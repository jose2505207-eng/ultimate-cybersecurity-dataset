"""Normalize OWASP GenAI Top 10 risks."""

from scripts.normalizers.base import AI_CAT, parse_owasp_genai, run_module


def main() -> None:
    """Run the OWASP GenAI Top 10 normalizer."""

    run_module(module="ai_security_owasp_genai_top10", source_dataset="owasp_genai_top10", source_type="cti_taxonomy", main_category=AI_CAT, license_name="CC-BY-SA-4.0", parser=parse_owasp_genai)


if __name__ == "__main__":
    main()
