"""Normalize local CAPEC attack-pattern XML or CSV exports."""

from scripts.normalizers.base import THREAT_CAT, parse_capec, run_module


def main() -> None:
    """Run the CAPEC normalizer."""

    run_module(module="cti_capec_attack_patterns", source_dataset="capec", source_type="cti_taxonomy", main_category=THREAT_CAT, license_name="CAPEC-MITRE", parser=parse_capec)


if __name__ == "__main__":
    main()
