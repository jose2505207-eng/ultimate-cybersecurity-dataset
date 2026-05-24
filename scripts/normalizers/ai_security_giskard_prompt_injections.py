"""Normalize Giskard prompt injection records."""

from scripts.normalizers.base import AI_CAT, parse_giskard, run_module


def main() -> None:
    """Run the Giskard prompt normalizer."""

    run_module(module="ai_security_giskard_prompt_injections", source_dataset="giskard_prompt_injections", source_type="prompt_text", main_category=AI_CAT, license_name="UNKNOWN", parser=parse_giskard)


if __name__ == "__main__":
    main()
