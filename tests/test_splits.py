from cyberdataset.build.build_silver import smoke_records
from cyberdataset.normalize import finalize_records
from cyberdataset.splitting import VALID_SPLITS, assign_split


def test_split_assignment_is_deterministic():
    assert assign_split("example:1") == assign_split("example:1")


def test_splits_are_valid():
    df = finalize_records(smoke_records())
    assert set(df["split"]).issubset(VALID_SPLITS)


def test_no_record_id_overlap_across_splits():
    df = finalize_records(smoke_records())
    split_sets = {split: set(group["record_id"]) for split, group in df.groupby("split")}
    seen = set()
    for record_ids in split_sets.values():
        assert seen.isdisjoint(record_ids)
        seen.update(record_ids)

