from cyberdataset.normalize import map_label


def test_vulnerable_mapping():
    mapped = map_label("vulnerable")
    assert mapped["label"] == "vulnerable"
    assert mapped["binary_label"] == 1


def test_benign_mapping():
    mapped = map_label("benign")
    assert mapped["label"] == "benign"
    assert mapped["binary_label"] == 0


def test_phishing_mapping():
    mapped = map_label("phishing")
    assert mapped["label"] == "phishing"
    assert mapped["attack_family"] == "Phishing"


def test_malware_mapping():
    mapped = map_label("malware")
    assert mapped["label"] == "malware"
    assert mapped["binary_label"] == 1


def test_cti_reference_mapping_has_null_binary_label():
    mapped = map_label("cti_reference")
    assert mapped["label"] == "cti_reference"
    assert mapped["binary_label"] is None


def test_prompt_attack_mapping():
    mapped = map_label("prompt_attack")
    assert mapped["label"] == "prompt_attack"
    assert mapped["attack_family"] == "AI Security"


def test_advisory_mapping_has_null_binary_label():
    mapped = map_label("advisory")
    assert mapped["label"] == "advisory"
    assert mapped["binary_label"] is None
