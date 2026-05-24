"""Normalize UNSW-NB15 flow CSVs from local CSV-only zip files."""

from scripts.normalizers.base import NETWORK_CAT, parse_unsw_nb15, run_module


def main() -> None:
    """Run the UNSW-NB15 normalizer."""

    run_module(module="network_unsw_nb15", source_dataset="unsw_nb15", source_type="network_flow", main_category=NETWORK_CAT, license_name="RESTRICTED:UNSW-Academic", parser=parse_unsw_nb15)


if __name__ == "__main__":
    main()
