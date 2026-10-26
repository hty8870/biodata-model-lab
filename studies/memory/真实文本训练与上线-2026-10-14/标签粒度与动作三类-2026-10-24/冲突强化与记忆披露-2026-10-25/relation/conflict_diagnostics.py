"""共享的因素签名审计与保守语义依赖簇；不从表面措辞推断语义。"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import conflict_common as K


def factor_key(factor: dict, setting: dict) -> str:
    import conflict_semantics as S
    enriched = S.with_setting(factor, setting)
    return K.sha256_json([enriched["subject"], enriched["slot"], enriched["object"],
                          enriched.get("referent") or "", sorted(S._applicability(enriched)),
                          sorted(S._allowed(enriched))])


def item_keys(spec: dict, item: dict, *, meaning: bool = False) -> dict:
    if meaning:
        import conflict_semantics as S
        setting = S.SETTINGS_BY_ID.get(spec["setting"])
    else:
        setting = None
    old = factor_key(item["old"], setting) if setting else K.sha256_json([spec["setting"], item["old"]])
    new = factor_key(item["new"], setting) if setting else K.sha256_json([spec["setting"], item["new"]])
    return {"old": old, "new": new, "pair": K.sha256_json([old, new])}


def semantic_twins(specs, *, meaning: bool = False) -> dict:
    entries = [(spec, item, item_keys(spec, item, meaning=meaning)) for spec in specs for item in spec["items"]]
    definition = ("classifier denotations: identity, applicability set and allowed values" if meaning
                  else "exact declared factor signatures")
    report = {"definition": definition + "; within counts are excess rows, not independent pairs",
              "within": {}, "cross": {}}
    for split in K.SPLITS:
        members = [(spec, item, keys) for spec, item, keys in entries if spec["split"] == split]
        report["within"][split] = {}
        for kind in ("pair", "old", "new"):
            groups = defaultdict(list)
            for _, item, keys in members:
                groups[keys[kind]].append(item["pair_id"])
            duplicates = [ids for ids in groups.values() if len(ids) > 1]
            report["within"][split][kind] = {
                "rows": len(members), "unique": len(groups),
                "excess_rows": sum(len(ids) - 1 for ids in duplicates),
                "groups": len(duplicates), "examples": duplicates[:5]}
    for left, right in combinations(K.SPLITS, 2):
        report["cross"][f"{left}|{right}"] = {}
        for name, sides in (("pair", ("pair",)), ("old", ("old",)),
                            ("new", ("new",)), ("any_side", ("old", "new"))):
            groups = {split: defaultdict(list) for split in (left, right)}
            for spec, item, keys in entries:
                if spec["split"] in groups:
                    for side in sides:
                        groups[spec["split"]][keys[side]].append(item["pair_id"])
            common = sorted(set(groups[left]) & set(groups[right]))
            report["cross"][f"{left}|{right}"][name] = {
                "count": len(common), "examples": [
                    {left: groups[left][key][:4], right: groups[right][key][:4]} for key in common[:5]]}
    report["cross_ok"] = all(block["count"] == 0 for pair in report["cross"].values()
                             for block in pair.values())
    report["strata"] = {}
    for split in K.SPLITS:
        selected = [(spec, item) for spec, item, _ in entries if spec["split"] == split]
        report["strata"][split] = {}
        for field in ("setting", "recipe"):
            strata = defaultdict(list)
            for spec, item in selected:
                strata[spec[field] if field == "setting" else item[field]].append((spec, item))
            report["strata"][split][field] = {
                key: {"rows": len(values), "families": len({s["family_id"] for s, _ in values}),
                      "coarse": dict(Counter(i["expected_coarse"] for _, i in values)),
                      "pools": dict(Counter(i["pool"] for _, i in values))}
                for key, values in sorted(strata.items())}
    report["novelty"] = {}
    for split in K.SPLITS:
        selected = [(item, keys) for spec, item, keys in entries if spec["split"] == split]
        pools = {}
        for pool in K.POOLS:
            values = [keys["pair"] for item, keys in selected if item["pool"] == pool]
            pools[pool] = {"rows": len(values), "unique_pairs": len(set(values)),
                           "excess_rows": len(values) - len(set(values))}
        control = {keys["pair"] for item, keys in selected if item["pool"] == K.POOL_CONTROL}
        cf = {keys["pair"] for item, keys in selected if item["pool"] == K.POOL_COUNTERFACTUAL}
        report["novelty"][split] = {"pools": pools, "control_cf_shared_pair_keys": len(control & cf)}
    return report


def semantic_clusters(rows, specs=()) -> dict:
    """按 family 和任一重复 side/pair 的连通分量合簇，不能把这些行视为独立样本。"""
    rows = list(rows)
    spec_items = {item["pair_id"]: (spec, item) for spec in specs for item in spec["items"]}
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        a, b = find(first), find(second)
        if a != b:
            parents[b] = a

    seen = {}
    mapped = 0
    for index, row in enumerate(rows):
        keys = [("family", row.get("family_id") or row.get("cluster_id") or row["id"])]
        if row["id"] in spec_items:
            mapped += 1
            signatures = item_keys(*spec_items[row["id"]], meaning=True)
            keys.extend(("factor_side", signatures[side]) for side in ("old", "new"))
            keys.append(("factor_pair", signatures["pair"]))
        elif row.get("pair"):
            keys.extend(("text_side", K.normalize_text(text)) for text in K.pair_sides(row["pair"]))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[find(index)].append(row["id"])
    names = {root: "semantic-" + K.sha256_json(sorted(ids))[:16] for root, ids in members.items()}
    return {"ids": [names[find(index)] for index in range(len(rows))],
            "clusters": len(members), "spec_mapped_rows": mapped,
            "spec_sha256": K.sha256_json(specs) if specs else None,
            "definition": "family plus shared old/new classifier denotations (identity/applicability/allowed values); unknown settings use exact factor signatures, unmapped rows exact normalized side text only",
            "independent_rows_assumed": False,
            "sizes": sorted(len(ids) for ids in members.values()),
            "cross_family_clusters": sum(len({rows[i].get("family_id") or rows[i]["id"]
                                               for i in range(len(rows)) if find(i) == root}) > 1
                                         for root in members)}
