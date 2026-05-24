"""Normalize local HackAPrompt records when present."""

from scripts.normalizers.base import AI_CAT, parse_hackaprompt, run_module


def main() -> None:
    """Run the HackAPrompt normalizer."""

    run_module(module="ai_security_hackaprompt", source_dataset="ai_security_prompt_injection", source_type="prompt_text", main_category=AI_CAT, license_name="UNKNOWN", parser=parse_hackaprompt)


if __name__ == "__main__":
    main()
