"""Normalize SARD Juliet C/C++ vulnerability test cases."""

from scripts.normalizers.base import VULN_CAT, parse_sard_juliet, run_module


def main() -> None:
    """Run the SARD Juliet normalizer."""

    run_module(
        module="vulnerable_code_sard_juliet",
        source_dataset="2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip",
        source_type="vulnerable_code",
        main_category=VULN_CAT,
        license_name="UNKNOWN",
        parser=parse_sard_juliet,
    )


if __name__ == "__main__":
    main()
