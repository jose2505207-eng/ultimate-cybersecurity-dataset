"""Normalize SmartBugs Curated Solidity vulnerability sources."""

from scripts.normalizers.base import CRYPTO_CAT, parse_smartbugs, run_module


def main() -> None:
    """Run the SmartBugs normalizer."""

    run_module(module="blockchain_smartbugs_curated", source_dataset="smartbugs_curated", source_type="smart_contract_code", main_category=CRYPTO_CAT, license_name="UNKNOWN", parser=parse_smartbugs)


if __name__ == "__main__":
    main()
