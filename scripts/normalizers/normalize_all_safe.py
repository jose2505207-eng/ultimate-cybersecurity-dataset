"""Safely run registered normalizers in subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC
from pathlib import Path

import pandas as pd

from scripts.normalizers.common import PROJECT_ROOT
from scripts.normalizers.registry import NORMALIZERS

DEFAULT_MEMORY_BYTES = 4 * 1024**3


def append_error(module: str, exc: BaseException | None, message: str) -> None:
    """Append structured error JSONL."""

    output = PROJECT_ROOT / "data" / "silver_normalized" / "normalization_errors.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "module": module,
        "error_class": type(exc).__name__ if exc else "SubprocessError",
        "message": message,
        "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]) if exc else "",
        "timestamp_utc": pd.Timestamp.now(tz=UTC).isoformat(),
    }
    with output.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def _resource_limiter(memory_bytes: int = DEFAULT_MEMORY_BYTES):
    """Return a subprocess preexec_fn that applies Linux resource caps."""

    if os.name != "posix":
        return None

    def _limit() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except Exception:
            pass

    return _limit


def _append_run_log(path: Path, payload: dict[str, object]) -> None:
    """Append one structured runner event."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _module_metadata(module: str) -> dict[str, object]:
    """Load module metadata if it exists."""

    meta_path = PROJECT_ROOT / "data" / "silver_normalized" / module / f"{module}_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--memory-bytes", type=int, default=DEFAULT_MEMORY_BYTES)
    args = parser.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rows = []
    run_ts = pd.Timestamp.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_log = PROJECT_ROOT / "logs" / "silver" / f"normalize_all_safe_{run_ts}.jsonl"
    for name, spec in sorted(NORMALIZERS.items(), key=lambda kv: (kv[1]["priority"], kv[0])):
        if only and name not in only:
            continue
        if args.priority and spec["priority"] != args.priority:
            continue
        cmd = [sys.executable, "-m", f"scripts.normalizers.{spec['module']}", "--input", str(PROJECT_ROOT / "data" / "bronze_raw" / spec["input"]), "--output", str(PROJECT_ROOT / "data" / "silver_normalized")]
        if args.force:
            cmd.append("--force")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.max_rows:
            cmd.extend(["--max-rows", str(args.max_rows)])
        started = time.time()
        event = {
            "module": name,
            "command": cmd,
            "dry_run": args.dry_run,
            "started_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        }
        _append_run_log(run_log, {**event, "event": "start"})
        try:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
                check=False,
                preexec_fn=_resource_limiter(args.memory_bytes),
            )
            meta = {} if args.dry_run else _module_metadata(name)
            status = str(meta.get("status") or ("ok" if proc.returncode == 0 else "failed"))
            if proc.returncode != 0:
                append_error(name, None, (proc.stderr or proc.stdout)[-2000:])
                status = "failed"
            row = {
                "module": name,
                "status": status,
                "rows": meta.get("row_count", "") if meta else "",
                "bytes": sum((meta.get("output_bytes") or {}).values()) if meta.get("output_bytes") else "",
                "duration_s": round(time.time() - started, 2),
            }
            rows.append(row)
            _append_run_log(
                run_log,
                {
                    **event,
                    "event": "finish",
                    "status": status,
                    "returncode": proc.returncode,
                    "duration_s": row["duration_s"],
                    "stdout_tail": proc.stdout[-2000:],
                    "stderr_tail": proc.stderr[-2000:],
                },
            )
        except subprocess.TimeoutExpired as exc:
            append_error(name, exc, f"timeout after {args.timeout_seconds}s")
            row = {"module": name, "status": "failed", "rows": "", "bytes": "", "duration_s": round(time.time() - started, 2)}
            rows.append(row)
            _append_run_log(run_log, {**event, "event": "timeout", "status": "failed", "duration_s": row["duration_s"]})
        except Exception as exc:
            append_error(name, exc, str(exc))
            row = {"module": name, "status": "failed", "rows": "", "bytes": "", "duration_s": round(time.time() - started, 2)}
            rows.append(row)
            _append_run_log(run_log, {**event, "event": "error", "status": "failed", "duration_s": row["duration_s"], "error": str(exc)})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
