"""Normalize OWASP crAPI local API challenge documentation."""

from scripts.normalizers.base import API_CAT, parse_owasp_crapi, run_module


def main() -> None:
    """Run the OWASP crAPI normalizer."""

    run_module(
        module="api_owasp_crapi",
        source_dataset="owasp_api_security",
        source_type="api_request",
        main_category=API_CAT,
        license_name="Apache-2.0",
        parser=parse_owasp_crapi,
    )


if __name__ == "__main__":
    main()
