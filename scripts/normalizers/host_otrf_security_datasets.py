"""Normalize OTRF Security Datasets local metadata."""

from scripts.normalizers.base import HOST_CAT, parse_otrf_security_datasets, run_module


def main() -> None:
    """Run the OTRF Security Datasets normalizer."""

    run_module(
        module="host_otrf_security_datasets",
        source_dataset="otrf_security_datasets",
        source_type="host_telemetry",
        main_category=HOST_CAT,
        license_name="UNKNOWN",
        parser=parse_otrf_security_datasets,
    )


if __name__ == "__main__":
    main()
