"""Normalize Lakera Gandalf prompt injection records."""

from scripts.normalizers.base import AI_CAT, parse_lakera, run_module


def main() -> None:
    """Run the Lakera Gandalf normalizer."""

    run_module(module="ai_security_lakera_gandalf", source_dataset="huggingface_ai_security", source_type="prompt_text", main_category=AI_CAT, license_name="UNKNOWN", parser=parse_lakera)


if __name__ == "__main__":
    main()
