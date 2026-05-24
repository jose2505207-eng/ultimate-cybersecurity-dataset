"""Plan silver modules for every inventoried bronze folder."""

from __future__ import annotations

import argparse

import pandas as pd

from scripts.normalizers.common import PROJECT_ROOT

MODULES = {
    "mitre_attack_stix": ("cti_mitre_attack_stix", 1),
    "github_advisory_database": ("supply_chain_github_advisory", 1),
    "osv": ("supply_chain_osv", 1),
    "nvd_cve": ("advisory_nvd_cve", 1),
    "phishtank": ("phishing_phishtank", 1),
    "phishing_balanced_urls": ("phishing_balanced_urls", 1),
    "owasp_genai_top10": ("ai_security_owasp_genai_top10", 1),
    "giskard_prompt_injections": ("ai_security_giskard_prompt_injections", 1),
    "huggingface_ai_security": ("ai_security_lakera_gandalf", 1),
    "huggingface": ("ai_security_huggingface_prompt_injection", 1),
    "ai_security_prompt_injection": ("ai_security_hackaprompt", 1),
    "smartbugs_curated": ("blockchain_smartbugs_curated", 2),
    "defihacklabs_incident_explorer": ("blockchain_defihacklabs_incidents", 2),
    "otrf_security_datasets": ("host_otrf_security_datasets", 3),
    "owasp_api_security": ("api_owasp_crapi", 3),
    "owasp_benchmark": ("web_owasp_benchmark", 3),
    "2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip": ("vulnerable_code_sard_juliet", 3),
}

IMPLEMENTED_MODULES = {module for module, _priority in MODULES.values()}


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    catalog = PROJECT_ROOT / "data" / "bronze_catalog"
    inv = pd.read_csv(catalog / "bronze_inventory.csv")
    rows = []
    for row in inv.to_dict("records"):
        module, priority = MODULES.get(row["dataset_folder"], (f"needs_review_{row['dataset_folder']}", 3))
        rows.append(
            {
                "dataset_folder": row["dataset_folder"],
                "planned_silver_module": module,
                "planned_category": row["main_category"],
                "normalizer_status": "implemented" if module in IMPLEMENTED_MODULES and not module.startswith("needs_review") else "not_started",
                "priority": priority,
                "blocker_reason": "" if module in IMPLEMENTED_MODULES else "deferred_priority",
                "estimated_rows": "",
                "requires_extraction": module == "supply_chain_osv",
            }
        )
    pd.DataFrame(rows).to_csv(catalog / "silver_normalization_plan.csv", index=False)


if __name__ == "__main__":
    main()
