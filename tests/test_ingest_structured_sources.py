import json

import pandas as pd

from cyberdataset.ingest import ingest_cicids2017, ingest_cisa_kev, ingest_nvd, ingest_phishtank, ingest_unsw_nb15, ingest_urlhaus
from cyberdataset.schema import validate_schema


def test_cicids2017_parser_maps_flow_labels(tmp_path):
    path = tmp_path / "flows.csv"
    path.write_text(
        "Flow Duration,Total Fwd Packets, Label\n"
        "10,3,BENIGN\n"
        "20,8,DDoS\n",
        encoding="utf-8",
    )

    df = ingest_cicids2017.normalize(ingest_cicids2017.load_raw(tmp_path))

    validate_schema(df)
    assert df.loc[0, "label"] == "benign"
    assert df.loc[1, "main_category"] == "denial_of_service"
    assert df.loc[1, "binary_label"] == 1


def test_unsw_nb15_parser_uses_attack_category(tmp_path):
    path = tmp_path / "unsw.csv"
    path.write_text(
        "dur,proto,attack_cat,label\n"
        "1,tcp,Normal,0\n"
        "2,tcp,Exploits,1\n",
        encoding="utf-8",
    )

    df = ingest_unsw_nb15.normalize(ingest_unsw_nb15.load_raw(tmp_path))

    validate_schema(df)
    assert df.loc[0, "label"] == "benign"
    assert df.loc[1, "attack_name"] == "Exploits"


def test_phishtank_parser_defangs_url(tmp_path):
    path = tmp_path / "phishtank.csv"
    path.write_text("phish_id,url\n42,https://bad.example/login\n", encoding="utf-8")

    df = ingest_phishtank.normalize(ingest_phishtank.load_raw(tmp_path))

    validate_schema(df)
    assert df.loc[0, "label"] == "phishing"
    assert "hxxp://" in df.loc[0, "raw_text_or_features"]


def test_urlhaus_parser_maps_malware_url(tmp_path):
    path = tmp_path / "urlhaus.csv"
    path.write_text("id,url,threat\n7,http://mal.example/payload,malware_download\n", encoding="utf-8")

    df = ingest_urlhaus.normalize(ingest_urlhaus.load_raw(tmp_path))

    validate_schema(df)
    assert df.loc[0, "label"] == "malware"
    assert df.loc[0, "main_category"] == "malware"


def test_nvd_parser_handles_api_json(tmp_path):
    payload = {
        "vulnerabilities": [
            {
                "cve": {
                    "id": "CVE-2099-0001",
                    "descriptions": [{"lang": "en", "value": "Example vulnerability advisory."}],
                    "weaknesses": [{"description": [{"lang": "en", "value": "CWE-79"}]}],
                },
                "metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "HIGH"}}]},
                "published": "2099-01-01T00:00:00.000",
            }
        ]
    }
    (tmp_path / "nvd.json").write_text(json.dumps(payload), encoding="utf-8")

    df = ingest_nvd.normalize(ingest_nvd.load_raw(tmp_path))

    validate_schema(df)
    assert df.loc[0, "label"] == "advisory"
    assert pd.isna(df.loc[0, "binary_label"])
    assert df.loc[0, "cve_id"] == "CVE-2099-0001"
    assert df.loc[0, "severity"] == "high"


def test_cisa_kev_parser_handles_catalog_json(tmp_path):
    payload = {
        "vulnerabilities": [
            {
                "cveID": "CVE-2099-0002",
                "vendorProject": "ExampleVendor",
                "product": "ExampleProduct",
                "vulnerabilityName": "Example KEV",
                "shortDescription": "Example catalog entry.",
            }
        ]
    }
    (tmp_path / "kev.json").write_text(json.dumps(payload), encoding="utf-8")

    df = ingest_cisa_kev.normalize(ingest_cisa_kev.load_raw(tmp_path))

    validate_schema(df)
    assert df.loc[0, "source_dataset"] == "CISA_KEV"
    assert df.loc[0, "cve_id"] == "CVE-2099-0002"
    assert df.loc[0, "severity"] == "critical"

