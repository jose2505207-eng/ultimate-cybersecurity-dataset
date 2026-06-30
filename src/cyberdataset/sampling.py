from __future__ import annotations

import pandas as pd

from cyberdataset.utils import config_path, load_yaml


def quota_sample(df: pd.DataFrame, *, max_rows: int | None = None, seed: int | None = None) -> pd.DataFrame:
    plan = load_yaml(config_path("sampling_plan.yaml"))
    max_rows = max_rows or int(plan["max_rows"])
    seed = seed if seed is not None else int(plan.get("random_seed", 42))
    quotas = plan.get("category_quotas", {})

    sampled_parts: list[pd.DataFrame] = []
    for category, group in df.groupby("main_category", dropna=False):
        quota = int(quotas.get(category, max_rows))
        take = min(len(group), quota)
        sampled_parts.append(group.sample(n=take, random_state=seed) if take < len(group) else group)

    if not sampled_parts:
        return df.head(0).copy()

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.sort_values("record_id").reset_index(drop=True)

