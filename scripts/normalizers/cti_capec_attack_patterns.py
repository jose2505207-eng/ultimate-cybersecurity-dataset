"""Placeholder CAPEC normalizer for local CAPEC drops."""

from scripts.normalizers.base import THREAT_CAT, parse_hackaprompt, run_module


def main() -> None:
    """Run a no-op compatible parser until CAPEC files are present."""

    run_module(module="cti_capec_attack_patterns", source_dataset="capec", source_type="cti_taxonomy", main_category=THREAT_CAT, license_name="CAPEC-MITRE", parser=parse_hackaprompt)


if __name__ == "__main__":
    main()
