"""Normalize OSV vulnerability records."""

from scripts.normalizers.base import SUPPLY_CAT, parse_osv, run_module


def main() -> None:
    """Run the OSV normalizer."""

    run_module(module="supply_chain_osv", source_dataset="osv", source_type="package_metadata", main_category=SUPPLY_CAT, license_name="CC-BY-4.0", parser=parse_osv)


if __name__ == "__main__":
    main()
