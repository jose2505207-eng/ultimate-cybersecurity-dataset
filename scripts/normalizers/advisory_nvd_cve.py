"""Normalize NVD CVE advisories."""

from scripts.normalizers.base import THREAT_CAT, parse_nvd, run_module


def main() -> None:
    """Run the NVD normalizer."""

    run_module(module="advisory_nvd_cve", source_dataset="nvd_cve", source_type="vulnerability_advisory", main_category=THREAT_CAT, license_name="NVD-Public-Domain", parser=parse_nvd)


if __name__ == "__main__":
    main()
