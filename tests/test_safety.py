from cyberdataset.safety import ensure_safe, is_safe_representation, redact_text


def test_redacts_secret_like_values():
    redacted = redact_text("token=abcdefghijklmnop")
    assert "[REDACTED]" in redacted
    assert "abcdefghijklmnop" not in redacted


def test_defangs_urls():
    redacted = ensure_safe("https://example.com/path")
    assert "hxxp://" in redacted
    assert "example[.]com" in redacted


def test_safe_feature_vector_passes():
    value = '{"duration": 1.2, "bytes_in": 128}'
    assert is_safe_representation(value)

