"""Normalize PhishTank URL records without requesting URLs."""

from scripts.normalizers.base import PHISH_CAT, parse_phishtank, run_module


def main() -> None:
    """Run the PhishTank normalizer."""

    run_module(module="phishing_phishtank", source_dataset="phishtank", source_type="phishing_url", main_category=PHISH_CAT, license_name="RESTRICTED:PhishTank", parser=parse_phishtank)


if __name__ == "__main__":
    main()
