"""Normalize local balanced phishing URL records."""

from scripts.normalizers.base import PHISH_CAT, parse_balanced_urls, run_module


def main() -> None:
    """Run the balanced URL normalizer."""

    run_module(module="phishing_balanced_urls", source_dataset="phishing_balanced_urls", source_type="phishing_url", main_category=PHISH_CAT, license_name="UNKNOWN", parser=parse_balanced_urls)


if __name__ == "__main__":
    main()
