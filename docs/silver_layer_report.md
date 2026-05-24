# Silver Layer Report

Generated: 2026-05-24T18:19:33.641122+00:00

## Module Status
| silver_module | status | row_count | license | license_compatibility | notes |
| --- | --- | --- | --- | --- | --- |
| advisory_nvd_cve | ok | 40704 | NVD-Public-Domain | permissive |  |
| ai_security_giskard_prompt_injections | ok | 35 | UNKNOWN | unknown |  |
| ai_security_hackaprompt | blocked | 0 | UNKNOWN | unknown | HackAPrompt bronze input contains README/metadata only; no local CSV/JSON/Parquet prompt records were found. |
| ai_security_huggingface_prompt_injection | ok | 5832 | UNKNOWN | unknown |  |
| ai_security_lakera_gandalf | ok | 1000 | UNKNOWN | unknown |  |
| ai_security_owasp_genai_top10 | ok | 10 | CC-BY-SA-4.0 | permissive |  |
| api_owasp_crapi | ok | 18 | Apache-2.0 | permissive |  |
| blockchain_defihacklabs_incidents | ok | 627 | UNKNOWN | unknown |  |
| blockchain_smartbugs_curated | ok | 143 | UNKNOWN | unknown |  |
| cti_capec_attack_patterns | blocked | 0 | CAPEC-MITRE | permissive | No CAPEC CSV/XML source is present under data/bronze_raw/capec. |
| cti_mitre_attack_stix | ok | 918 | Apache-2.0 | permissive |  |
| host_otrf_security_datasets | ok | 113 | UNKNOWN | unknown |  |
| malware_cic_malmem_2022 | ok | 350 | RESTRICTED:CIC-Academic | restricted_academic |  |
| network_unsw_nb15 | ok | 50000 | RESTRICTED:UNSW-Academic | restricted_academic |  |
| phishing_balanced_urls | ok | 50000 | UNKNOWN | unknown |  |
| phishing_phishtank | blocked | 0 | RESTRICTED:PhishTank | restricted_other | Local PhishTank file is not a valid CSV export; it contains a rate-limit response instead of URL records. |
| supply_chain_github_advisory | ok | 50000 | CC-BY-4.0 | permissive |  |
| supply_chain_osv | ok | 49827 | CC-BY-4.0 | permissive |  |
| vulnerable_code_sard_juliet | ok | 50000 | UNKNOWN | unknown |  |
| web_owasp_benchmark | ok | 2740 | GPL-2.0-only | restricted_other |  |

## Succeeded Modules
| silver_module | row_count | license |
| --- | --- | --- |
| advisory_nvd_cve | 40704 | NVD-Public-Domain |
| ai_security_giskard_prompt_injections | 35 | UNKNOWN |
| ai_security_huggingface_prompt_injection | 5832 | UNKNOWN |
| ai_security_lakera_gandalf | 1000 | UNKNOWN |
| ai_security_owasp_genai_top10 | 10 | CC-BY-SA-4.0 |
| api_owasp_crapi | 18 | Apache-2.0 |
| blockchain_defihacklabs_incidents | 627 | UNKNOWN |
| blockchain_smartbugs_curated | 143 | UNKNOWN |
| cti_mitre_attack_stix | 918 | Apache-2.0 |
| host_otrf_security_datasets | 113 | UNKNOWN |
| malware_cic_malmem_2022 | 350 | RESTRICTED:CIC-Academic |
| network_unsw_nb15 | 50000 | RESTRICTED:UNSW-Academic |
| phishing_balanced_urls | 50000 | UNKNOWN |
| supply_chain_github_advisory | 50000 | CC-BY-4.0 |
| supply_chain_osv | 49827 | CC-BY-4.0 |
| vulnerable_code_sard_juliet | 50000 | UNKNOWN |
| web_owasp_benchmark | 2740 | GPL-2.0-only |

## Skipped or Blocked Modules
| silver_module | status | notes |
| --- | --- | --- |
| ai_security_hackaprompt | blocked | HackAPrompt bronze input contains README/metadata only; no local CSV/JSON/Parquet prompt records were found. |
| cti_capec_attack_patterns | blocked | No CAPEC CSV/XML source is present under data/bronze_raw/capec. |
| phishing_phishtank | blocked | Local PhishTank file is not a valid CSV export; it contains a rate-limit response instead of URL records. |

## Rows by Category
| main_category | row_count |
| --- | --- |
| AI, LLM & ML Security | 6877 |
| API Security | 18 |
| Cryptocurrency & Blockchain Attacks | 770 |
| Endpoint, Host & Windows/Sysmon Telemetry | 113 |
| Malware & PE/Memory Features | 350 |
| Network Intrusion & Traffic Attacks | 50000 |
| Phishing, Social Engineering & Fraud | 50000 |
| Supply Chain & Open Source Package Security | 99827 |
| Threat Intelligence, CVE, Advisory & Taxonomy | 41622 |
| Vulnerable Code & Software Weaknesses | 50000 |
| Web Application Security | 2740 |

## Rows by Module
| silver_module | row_count |
| --- | --- |
| network_unsw_nb15 | 50000 |
| phishing_balanced_urls | 50000 |
| vulnerable_code_sard_juliet | 50000 |
| supply_chain_github_advisory | 50000 |
| supply_chain_osv | 49827 |
| advisory_nvd_cve | 40704 |
| ai_security_huggingface_prompt_injection | 5832 |
| web_owasp_benchmark | 2740 |
| ai_security_lakera_gandalf | 1000 |
| cti_mitre_attack_stix | 918 |
| blockchain_defihacklabs_incidents | 627 |
| malware_cic_malmem_2022 | 350 |
| blockchain_smartbugs_curated | 143 |
| host_otrf_security_datasets | 113 |
| ai_security_giskard_prompt_injections | 35 |
| api_owasp_crapi | 18 |
| ai_security_owasp_genai_top10 | 10 |
| ai_security_hackaprompt | 0 |
| cti_capec_attack_patterns | 0 |
| phishing_phishtank | 0 |

## Label Distribution
- advisory_nvd_cve: {"vulnerability_advisory": 40704}
- ai_security_giskard_prompt_injections: {"jailbreak_prompt": 20, "malicious_prompt": 15}
- ai_security_hackaprompt: {}
- ai_security_huggingface_prompt_injection: {"malicious_prompt": 5832}
- ai_security_lakera_gandalf: {"malicious_prompt": 1000}
- ai_security_owasp_genai_top10: {"ai_security_risk": 10}
- api_owasp_crapi: {"attack_pattern": 18}
- blockchain_defihacklabs_incidents: {"exploit_incident": 627}
- blockchain_smartbugs_curated: {"vulnerable_code": 143}
- cti_capec_attack_patterns: {}
- cti_mitre_attack_stix: {"attack_technique": 918}
- host_otrf_security_datasets: {"attack_technique": 113}
- malware_cic_malmem_2022: {"malicious": 350}
- network_unsw_nb15: {"benign": 20520, "intrusion": 29480}
- phishing_balanced_urls: {"benign_url": 26066, "phishing": 23934}
- phishing_phishtank: {}
- supply_chain_github_advisory: {"vulnerable_dependency": 50000}
- supply_chain_osv: {"vulnerable_dependency": 49827}
- vulnerable_code_sard_juliet: {"non_vulnerable_code": 2782, "vulnerable_code": 47218}
- web_owasp_benchmark: {"non_vulnerable_code": 1325, "vulnerable_code": 1415}

## Silver Size on Disk
- Parquet bytes: 116008648
- CSV.GZ bytes: 68209855

## License Summary
- nvd_cve: NVD-Public-Domain (permissive)
- giskard_prompt_injections: UNKNOWN (unknown)
- ai_security_prompt_injection: UNKNOWN (unknown)
- huggingface: UNKNOWN (unknown)
- huggingface_ai_security: UNKNOWN (unknown)
- owasp_genai_top10: CC-BY-SA-4.0 (permissive)
- owasp_api_security: Apache-2.0 (permissive)
- defihacklabs_incident_explorer: UNKNOWN (unknown)
- smartbugs_curated: UNKNOWN (unknown)
- capec: CAPEC-MITRE (permissive)
- mitre_attack_stix: Apache-2.0 (permissive)
- otrf_security_datasets: UNKNOWN (unknown)
- cic_malmem_2022: RESTRICTED:CIC-Academic (restricted_academic) EXCLUDED FROM PUBLIC RELEASE
- unsw_nb15: RESTRICTED:UNSW-Academic (restricted_academic) EXCLUDED FROM PUBLIC RELEASE
- phishing_balanced_urls: UNKNOWN (unknown)
- phishtank: RESTRICTED:PhishTank (restricted_other) EXCLUDED FROM PUBLIC RELEASE
- github_advisory_database: CC-BY-4.0 (permissive)
- osv: CC-BY-4.0 (permissive)
- 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip: UNKNOWN (unknown)
- owasp_benchmark: GPL-2.0-only (restricted_other) EXCLUDED FROM PUBLIC RELEASE

## Cross-Source CVE Overlap
- Overlapping CVE rows: 38020

## Failures, Skips, and Blockers
### Failed
No failed modules in manifest.

### Skipped or Blocked
| silver_module | status | notes |
| --- | --- | --- |
| ai_security_hackaprompt | blocked | HackAPrompt bronze input contains README/metadata only; no local CSV/JSON/Parquet prompt records were found. |
| cti_capec_attack_patterns | blocked | No CAPEC CSV/XML source is present under data/bronze_raw/capec. |
| phishing_phishtank | blocked | Local PhishTank file is not a valid CSV export; it contains a rate-limit response instead of URL records. |

### Error Log Tail
No errors logged.

## Next Recommended Normalizers
- Remaining Priority 2: malware_bodmas, network_cic_ids_2017, supply_chain_datadog_malicious_packages.
- Remaining Priority 3: auth_lanl_authentication, iot_iot23, ics_swat_wadi_epic_batadal, insider_cert_insider_threat.

## Open Data Quality Issues
- CAPEC source is missing locally; module is blocked until a CSV/XML source is added.
- HackAPrompt source has README/metadata only; module is blocked until local prompt records are added.
- PhishTank local file is a rate-limit response, not a CSV export; module is blocked until a valid local export is added.
- Review preflight loose files, suspicious binaries/scripts, and incomplete download records before benchmark work.
