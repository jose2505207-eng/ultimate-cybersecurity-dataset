from cyberdataset.inventory import scan_bronze


def test_scan_bronze_reports_files_and_missing_metadata(tmp_path):
    source = tmp_path / "CICIDS2017"
    source.mkdir()
    (source / "flows.csv").write_text("Label,Flow Duration\nBENIGN,1\n", encoding="utf-8")

    report = scan_bronze(tmp_path)

    assert report[0]["source"] == "CICIDS2017"
    assert report[0]["file_count"] == 1
    assert report[0]["formats"] == ["csv"]
    assert "README.txt" in report[0]["missing_metadata"]

