"""Normalize DeFiHackLabs incident metadata and local RCA text."""

from scripts.normalizers.base import CRYPTO_CAT, parse_defihacklabs, run_module


def main() -> None:
    """Run the DeFiHackLabs incident normalizer."""

    run_module(module="blockchain_defihacklabs_incidents", source_dataset="defihacklabs_incident_explorer", source_type="defi_incident", main_category=CRYPTO_CAT, license_name="UNKNOWN", parser=parse_defihacklabs)


if __name__ == "__main__":
    main()
