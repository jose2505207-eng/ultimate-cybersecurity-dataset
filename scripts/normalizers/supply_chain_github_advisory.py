"""Normalize GitHub Advisory Database records."""

from scripts.normalizers.base import SUPPLY_CAT, parse_ghsa, run_module


def main() -> None:
    """Run the GHSA normalizer."""

    run_module(module="supply_chain_github_advisory", source_dataset="github_advisory_database", source_type="package_metadata", main_category=SUPPLY_CAT, license_name="CC-BY-4.0", parser=parse_ghsa)


if __name__ == "__main__":
    main()
