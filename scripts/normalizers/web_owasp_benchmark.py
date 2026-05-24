"""Normalize OWASP Benchmark Java expected-result sources."""

from scripts.normalizers.base import WEB_CAT, parse_owasp_benchmark, run_module


def main() -> None:
    """Run the OWASP Benchmark normalizer."""

    run_module(
        module="web_owasp_benchmark",
        source_dataset="owasp_benchmark",
        source_type="web_app_request",
        main_category=WEB_CAT,
        license_name="GPL-2.0-only",
        parser=parse_owasp_benchmark,
    )


if __name__ == "__main__":
    main()
