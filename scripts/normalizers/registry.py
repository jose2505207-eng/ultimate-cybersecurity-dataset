"""Registry of silver normalizers."""

NORMALIZERS = {
    "cti_mitre_attack_stix": {"module": "cti_mitre_attack_stix", "priority": 1, "category": "Threat Intelligence, CVE, Advisory & Taxonomy", "input": "mitre_attack_stix"},
    "cti_capec_attack_patterns": {"module": "cti_capec_attack_patterns", "priority": 1, "category": "Threat Intelligence, CVE, Advisory & Taxonomy", "input": "capec"},
    "advisory_nvd_cve": {"module": "advisory_nvd_cve", "priority": 1, "category": "Threat Intelligence, CVE, Advisory & Taxonomy", "input": "nvd_cve"},
    "supply_chain_osv": {"module": "supply_chain_osv", "priority": 1, "category": "Supply Chain & Open Source Package Security", "input": "osv"},
    "supply_chain_github_advisory": {"module": "supply_chain_github_advisory", "priority": 1, "category": "Supply Chain & Open Source Package Security", "input": "github_advisory_database"},
    "phishing_phishtank": {"module": "phishing_phishtank", "priority": 1, "category": "Phishing, Social Engineering & Fraud", "input": "phishtank"},
    "phishing_balanced_urls": {"module": "phishing_balanced_urls", "priority": 1, "category": "Phishing, Social Engineering & Fraud", "input": "phishing_balanced_urls"},
    "ai_security_owasp_genai_top10": {"module": "ai_security_owasp_genai_top10", "priority": 1, "category": "AI, LLM & ML Security", "input": "owasp_genai_top10"},
    "ai_security_hackaprompt": {"module": "ai_security_hackaprompt", "priority": 1, "category": "AI, LLM & ML Security", "input": "ai_security_prompt_injection"},
    "ai_security_lakera_gandalf": {"module": "ai_security_lakera_gandalf", "priority": 1, "category": "AI, LLM & ML Security", "input": "huggingface_ai_security"},
    "ai_security_giskard_prompt_injections": {"module": "ai_security_giskard_prompt_injections", "priority": 1, "category": "AI, LLM & ML Security", "input": "giskard_prompt_injections"},
    "ai_security_huggingface_prompt_injection": {"module": "ai_security_huggingface_prompt_injection", "priority": 1, "category": "AI, LLM & ML Security", "input": "huggingface"},
    "blockchain_smartbugs_curated": {"module": "blockchain_smartbugs_curated", "priority": 2, "category": "Cryptocurrency & Blockchain Attacks", "input": "smartbugs_curated"},
    "blockchain_defihacklabs_incidents": {"module": "blockchain_defihacklabs_incidents", "priority": 2, "category": "Cryptocurrency & Blockchain Attacks", "input": "defihacklabs_incident_explorer"},
    "malware_cic_malmem_2022": {"module": "malware_cic_malmem_2022", "priority": 2, "category": "Malware & PE/Memory Features", "input": "output2.csv"},
    "network_unsw_nb15": {"module": "network_unsw_nb15", "priority": 2, "category": "Network Intrusion & Traffic Attacks", "input": "OneDrive_1_5-23-2026.zip"},
    "vulnerable_code_sard_juliet": {"module": "vulnerable_code_sard_juliet", "priority": 3, "category": "Vulnerable Code & Software Weaknesses", "input": "2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip"},
    "web_owasp_benchmark": {"module": "web_owasp_benchmark", "priority": 3, "category": "Web Application Security", "input": "owasp_benchmark"},
    "api_owasp_crapi": {"module": "api_owasp_crapi", "priority": 3, "category": "API Security", "input": "owasp_api_security"},
    "host_otrf_security_datasets": {"module": "host_otrf_security_datasets", "priority": 3, "category": "Endpoint, Host & Windows/Sysmon Telemetry", "input": "otrf_security_datasets"},
}
