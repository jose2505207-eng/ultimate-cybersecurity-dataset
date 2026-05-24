import pytest

from cyberdataset.build.build_silver import smoke_records
from cyberdataset.normalize import finalize_records
from cyberdataset.schema import CANONICAL_COLUMNS, DatasetValidationError, validate_schema


def test_smoke_records_match_schema():
    df = finalize_records(smoke_records())
    assert list(df.columns) == CANONICAL_COLUMNS
    validate_schema(df)
    assert df["record_id"].is_unique


def test_invalid_source_type_fails():
    df = finalize_records(smoke_records())
    df.loc[0, "source_type"] = "packet_capture"
    with pytest.raises(DatasetValidationError):
        validate_schema(df)


def test_duplicate_record_id_fails():
    df = finalize_records(smoke_records())
    df.loc[1, "record_id"] = df.loc[0, "record_id"]
    with pytest.raises(DatasetValidationError):
        validate_schema(df)

