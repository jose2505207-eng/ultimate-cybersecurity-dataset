"""Normalize MITRE ATT&CK STIX attack-patterns."""

from scripts.normalizers.base import THREAT_CAT, parse_mitre, run_module


def main() -> None:
    """Run the MITRE ATT&CK normalizer."""

    run_module(module="cti_mitre_attack_stix", source_dataset="mitre_attack_stix", source_type="cti_taxonomy", main_category=THREAT_CAT, license_name="Apache-2.0", parser=parse_mitre)


if __name__ == "__main__":
    main()
