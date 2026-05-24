"""Normalize local Hugging Face prompt injection records."""

from scripts.normalizers.base import AI_CAT, parse_hf_prompt, run_module


def main() -> None:
    """Run the Hugging Face prompt normalizer."""

    run_module(module="ai_security_huggingface_prompt_injection", source_dataset="huggingface", source_type="prompt_text", main_category=AI_CAT, license_name="UNKNOWN", parser=parse_hf_prompt)


if __name__ == "__main__":
    main()
