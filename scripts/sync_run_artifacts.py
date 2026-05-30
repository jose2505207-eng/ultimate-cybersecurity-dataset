"""Mirror small QLoRA run artifacts into tracked reports and push them."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "run"


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_git(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {
        "args": ["git", *args],
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def copy_if_exists(src: Path, dst: Path, copied: list[Path]) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)


def copy_eval_reports(eval_dir: Path, dst_dir: Path, copied: list[Path]) -> None:
    if not eval_dir.exists():
        return
    report_names = {
        "evaluation_summary.json",
        "model_comparison_metrics.csv",
        "evaluation_results.json",
        "evaluation_results.csv",
    }
    for path in sorted(eval_dir.iterdir()):
        if path.is_file() and (path.name in report_names or path.name.startswith("metrics_")):
            copy_if_exists(path, dst_dir / path.name, copied)


def write_status(run_dir: Path, eval_dir: Path | None, report_dir: Path, run_name: str, copied: list[Path]) -> None:
    train_metrics_path = run_dir / "train_metrics.json"
    eval_summary_path = eval_dir / "evaluation_summary.json" if eval_dir else None
    metrics: dict[str, Any] = {}
    if train_metrics_path.exists():
        try:
            metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    status = "completed" if train_metrics_path.exists() and (run_dir / "final_adapter" / "adapter_config.json").exists() else "running_or_interrupted"
    lines = [
        "# Training Status",
        "",
        f"Last updated: {now_utc()}",
        "",
        "## Run",
        "",
        f"- Status: {status}",
        f"- Run name: `{run_name}`",
        f"- Output dir: `{run_dir}`",
        f"- Report dir: `{report_dir}`",
        f"- Final adapter present: {(run_dir / 'final_adapter' / 'adapter_config.json').exists()}",
        f"- Train metrics present: {train_metrics_path.exists()}",
    ]
    if metrics:
        lines.extend(
            [
                "",
                "## Final/Latest Metrics",
                "",
                f"- Train loss: {metrics.get('train_loss')}",
                f"- Runtime seconds: {metrics.get('train_runtime')}",
                f"- Tokens/sec: {metrics.get('tokens_per_second')}",
                f"- Peak VRAM allocated MB: {metrics.get('vram_allocated_mb')}",
                f"- Peak VRAM reserved MB: {metrics.get('vram_reserved_mb')}",
                f"- Final adapter: `{metrics.get('final_adapter')}`",
                f"- Checkpoint dir: `{metrics.get('checkpoint_dir')}`",
            ]
        )
    if eval_summary_path is not None:
        lines.extend(
            [
                "",
                "## Evaluation",
                "",
                f"- Evaluation dir: `{eval_dir}`",
                f"- Evaluation summary present: {eval_summary_path.exists()}",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifact Policy",
            "",
            "- Heavy adapter checkpoints remain local under ignored `outputs/` paths.",
            "- Checkpoint metadata, logs, evaluation reports, and this status file are intended for git sync.",
        ]
    )
    root_status = PROJECT_ROOT / "TRAINING_STATUS.md"
    report_status = report_dir / "TRAINING_STATUS.md"
    root_status.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_status.parent.mkdir(parents=True, exist_ok=True)
    report_status.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.extend([root_status, report_status])


def commit_and_push(paths: list[Path], message: str, remote: str, branch: str, push: bool) -> dict[str, Any]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return {"status": "no_paths", "paths": []}
    rel_paths = [repo_relative(path) for path in existing]
    add = run_git(["add", "--force", "--", *rel_paths])
    diff = run_git(["diff", "--cached", "--quiet", "--", *rel_paths])
    result: dict[str, Any] = {"paths": rel_paths, "add": add}
    if diff["returncode"] == 0:
        result["status"] = "no_changes"
        return result
    commit = run_git(["commit", "-m", message, "--", *rel_paths])
    result["commit"] = commit
    if commit["returncode"] != 0:
        result["status"] = "commit_failed"
        return result
    if push:
        refspec = f"HEAD:{branch}" if branch else "HEAD"
        push_result = run_git(["push", remote, refspec])
        result["push"] = push_result
        result["status"] = "pushed" if push_result["returncode"] == 0 else "push_failed"
    else:
        result["status"] = "committed_no_push"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--message", default="chore: sync qwen32b run artifacts")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    run_name = safe_slug(args.run_name or args.run_dir.name)
    report_dir = PROJECT_ROOT / args.report_dir if not args.report_dir.is_absolute() else args.report_dir
    logs_dir = report_dir / "logs"
    metrics_dir = report_dir / "metrics"
    eval_dst = report_dir / "evaluation"
    copied: list[Path] = []

    copy_if_exists(args.run_dir / "cloud_train.log", logs_dir / "cloud_train.log", copied)
    copy_if_exists(args.run_dir / "cloud_eval_adapter.log", logs_dir / "cloud_eval_adapter.log", copied)
    copy_if_exists(args.run_dir / "train_events.jsonl", logs_dir / "train_events.jsonl", copied)
    copy_if_exists(args.run_dir / "prepare_summary.json", metrics_dir / "prepare_summary.json", copied)
    copy_if_exists(args.run_dir / "train_metrics.json", metrics_dir / "train_metrics.json", copied)
    if args.eval_dir is not None:
        copy_eval_reports(args.eval_dir, eval_dst, copied)

    manifest = {
        "created_at": now_utc(),
        "run_name": run_name,
        "run_dir": str(args.run_dir),
        "eval_dir": str(args.eval_dir) if args.eval_dir else None,
        "report_dir": str(report_dir),
        "copied": [repo_relative(path) for path in copied],
    }
    manifest_path = report_dir / "artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    copied.append(manifest_path)
    write_status(args.run_dir, args.eval_dir, report_dir, run_name, copied)

    sync = commit_and_push(copied, args.message, args.remote, args.branch, push=not args.no_push)
    print(json.dumps({"manifest": manifest, "git": sync}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
