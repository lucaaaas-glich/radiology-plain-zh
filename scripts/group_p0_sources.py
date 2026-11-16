"""Audit local input-only textual families and remaining validation sources.

No model loads, network operations, new output inspection or final selection.
Only hashes/counts are exported; containment does not union report components.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.build_exposure_manifest import CANONICAL, fingerprint, normalize


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def shingles(text, size=5):
    """Exactly the historical V3 casefolded word n-gram set definition."""
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


def numeric_template(text, settings):
    normalized = normalize(text)
    if (len(normalized) > settings["max_normalized_chars"] or
            len(re.findall(r"\w+", normalized)) > settings["max_word_tokens"]):
        return None
    value, count = re.subn(settings["number_pattern"], settings["replacement"], normalized)
    if not count and settings["require_at_least_one_number"]:
        return None
    # All punctuation, negation, anatomical words and units remain unchanged.
    return value


def near_pairs(sets, numerator=4, denominator=5, counters=None):
    """Yield all exact-threshold matches, using a complete prefix candidate join.

    The size bound is |B| >= t|A| for a preceding, shorter B. Any match has
    overlap >= ceil(t|A|) and >= ceil(t|B|). Prefixes of length
    |X|-ceil(t|X|)+1 under a shared total order must therefore intersect.
    Final verification uses integer arithmetic, including the equality boundary.
    """
    if not 0 < numerator <= denominator:
        raise ValueError("Invalid Jaccard threshold")
    counts = counters if counters is not None else {}
    counts.update(candidate_pairs_checked=0, matching_pairs=0)
    frequencies = Counter(token for values in sets.values() for token in values)
    # Integer IDs or tuples are both ordered deterministically after frequency.
    order = {token: index for index, token in enumerate(
        sorted(frequencies, key=lambda token: (frequencies[token], token)))}
    postings = defaultdict(list)
    for key in sorted(sets, key=lambda k: (len(sets[k]), k)):
        values = sets[key]
        size = len(values)
        if not size:
            continue
        prefix_size = size - (numerator * size + denominator - 1) // denominator + 1
        prefix = sorted(values, key=order.__getitem__)[:prefix_size]
        candidates = set()
        for token in prefix:
            for previous in postings[token]:
                if len(sets[previous]) * denominator >= numerator * size:
                    candidates.add(previous)
        for previous in sorted(candidates):
            other = sets[previous]
            counts["candidate_pairs_checked"] += 1
            intersection = len(values & other)
            union = size + len(other) - intersection
            if denominator * intersection >= numerator * union:
                counts["matching_pairs"] += 1
                yield previous, key, intersection, union
        for token in prefix:
            postings[token].append(key)


def build_group_map(keys, edges):
    """Connected components with a stable minimum-hash representative."""
    parent = {key: key for key in keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for edge in edges:
        if edge["type"] not in {"exact", "near", "template"}:
            continue
        a, b = find(edge["left"]), find(edge["right"])
        parent[max(a, b)] = min(a, b)
    return {key: find(key) for key in sorted(parent)}


def find_fragment_parents(texts, fragment_hashes, full_report_hashes):
    """Return possible parents only; never assert provenance or union parents."""
    normalized = {key: normalize(value) for key, value in texts.items()}
    links = {}
    for fragment in sorted(fragment_hashes):
        text = normalized[fragment]
        pattern = re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)")
        links[fragment] = sorted(key for key in full_report_hashes
                                 if key != fragment and len(normalized[key]) >= len(text)
                                 and pattern.search(normalized[key]))
    return links


def read_inputs(root, config):
    """Recover registered source fields, never generated predictions/targets."""
    manifest_path = config["exposure_manifest"]
    raw = (root / manifest_path).read_bytes()
    if sha256(raw) != config["exposure_manifest_sha256"]:
        raise ValueError("Exposure manifest SHA mismatch")
    manifest = json.loads(raw)
    sources = {manifest_path: {"sha256": sha256(raw), "bytes": len(raw)}}
    texts = {}
    v1_splits = defaultdict(set)

    def read(path):
        target = (root / path).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("Source outside project")
        value = target.read_bytes()
        expected = manifest["sources"].get(path, {}).get("file_sha256")
        if not expected or sha256(value) != expected:
            raise ValueError("Registered input source SHA mismatch: " + path)
        sources[path] = {"sha256": sha256(value), "bytes": len(value)}
        return value

    def add(text):
        if not isinstance(text, str) or not normalize(text):
            raise ValueError("Empty or invalid registered input")
        key = fingerprint(text)
        texts[key] = text
        return key

    for path, (_, kind, _) in CANONICAL.items():
        for line in read(path).decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = add(row["source" if kind == "diagnostic_fragment" else "report"])
            if path.startswith("data/pilot_v1/"):
                v1_splits[Path(path).stem].add(key)
    for path, variable in (("scripts/audit_v2_guard.py", "PAIRS"),
                           ("scripts/smoke_gpu.py", "row")):
        tree = ast.parse(read(path))
        value = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                         and target.id == variable for target in node.targets))
        for text in ([pair[2] for pair in value] if variable == "PAIRS" else [value["report"]]):
            add(text)
    for path in ("examples/中文报告.txt", "examples/英文报告.txt"):
        add(read(path).decode("utf-8"))
    # These three old synthetic control inputs have no separate input file.
    # Only the source field is accessed; no prediction/output field is inspected.
    controls = json.loads(read("results/v2/dev_nllb600_probe/positive_controls.json"))
    for row in controls["cases"]:
        add(row["source"])
    historical = {row["report_hash"] for row in manifest["reports"]}
    if (set(texts) != historical or
            len(historical) != config["expected_historical_unique_hashes"]):
        raise ValueError("Historical source reconstruction mismatch")
    return texts, manifest, sources, dict(v1_splits)


def split_conflicts(groups, v1_splits, edges):
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    result = {}
    for first, second in pairs:
        a, b = v1_splits[first], v1_splits[second]
        ga, gb = Counter(groups[key] for key in a), Counter(groups[key] for key in b)
        common = set(ga) & set(gb)
        direct = Counter(edge["type"] for edge in edges
                         if ((edge["left"] in a and edge["right"] in b) or
                             (edge["right"] in a and edge["left"] in b)))
        result[first + "__" + second] = {
            "shared_exact_hashes": len(a & b), "shared_components": len(common),
            "first_split_reports_in_shared_components": sum(ga[g] for g in common),
            "second_split_reports_in_shared_components": sum(gb[g] for g in common),
            "cross_split_report_pairs_in_shared_components": sum(ga[g] * gb[g] for g in common),
            "direct_edges_by_type": dict(sorted(direct.items())),
        }
    return result


def audit(root, config, config_sha):
    started = time.monotonic()
    texts, manifest, sources, v1_splits = read_inputs(root, config)
    historical = set(texts)
    # Importing pyarrow is CPU-only; read exactly these two source columns.
    import pyarrow.parquet as pq
    source_config = config["evaluation_source"]
    source_path = root / source_config["parquet"]
    raw = source_path.read_bytes()
    if sha256(raw) != source_config["expected_sha256"]:
        raise ValueError("Validation parquet SHA mismatch")
    sources[source_config["parquet"]] = {"sha256": sha256(raw), "bytes": len(raw)}
    del raw
    rows = pq.read_table(source_path, columns=source_config["read_columns"]).to_pylist()
    if len(rows) != source_config["expected_rows"]:
        raise ValueError("Validation parquet row-count mismatch")
    candidate_rows = []
    for number, row in enumerate(rows, 1):
        text = row["radiology_report"]
        if not isinstance(text, str) or not normalize(text):
            raise ValueError("Empty validation input at row " + str(number))
        key = fingerprint(text)
        texts.setdefault(key, text)
        candidate_rows.append({"row_number": number, "report_hash": key,
                               "dataset_source": str(row["source"]),
                               "normalized_characters": len(normalize(text))})
    del rows
    candidates = {row["report_hash"] for row in candidate_rows}
    occurrences = defaultdict(list)
    for key in historical:
        occurrences[key].append("historical_manifest")
    for row in candidate_rows:
        occurrences[row["report_hash"]].append("validation_row:" + str(row["row_number"]))
    edges = []
    for key, refs in sorted(occurrences.items()):
        for ref in refs[1:]:
            edges.append({"type": "exact", "left": key, "right": key,
                          "first_occurrence": refs[0], "second_occurrence": ref})
    # Intern n-grams as integer IDs for bounded CPU memory; no hash collisions.
    vocabulary = {}
    sets = {}
    for key, text in sorted(texts.items()):
        values = set()
        for token in sorted(shingles(text, config["near_duplicate"]["word_ngram_size"])):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
            values.add(vocabulary[token])
        sets[key] = values
    vocabulary_size = len(vocabulary)
    del vocabulary
    near_settings = config["near_duplicate"]
    join_counts = {}
    for first, second, intersection, union in near_pairs(
            sets, near_settings["jaccard_numerator"], near_settings["jaccard_denominator"],
            join_counts):
        edges.append({"type": "near", "left": first, "right": second,
                      "intersection": intersection, "union": union})
    empty_sets = sum(not value for value in sets.values())
    del sets
    templates = defaultdict(list)
    for key, text in sorted(texts.items()):
        value = numeric_template(text, config["short_numeric_template"])
        if value is not None:
            templates[value].append(key)
    template_families = []
    for value, members in sorted(templates.items()):
        if len(members) > 1:
            template_families.append({"template_sha256": sha256(value.encode("utf-8")),
                                      "members": members})
            # A star gives the identical component without quadratic edge storage.
            for member in members[1:]:
                edges.append({"type": "template", "left": members[0], "right": member,
                              "template_sha256": sha256(value.encode("utf-8"))})
    groups = build_group_map(texts, edges)
    fragments = {row["report_hash"] for row in manifest["reports"]
                 if "diagnostic_fragment" in row["source_types"]}
    full_reports = candidates | {row["report_hash"] for row in manifest["reports"]
                                if {"real_report", "synthetic_report"} & set(row["source_types"])}
    fragment_links = find_fragment_parents(texts, fragments, full_reports)
    parent_hashes = set().union(*(set(values) for values in fragment_links.values()))
    historical_groups = {groups[key] for key in historical}
    parent_groups = {groups[key] for key in parent_hashes}
    excluded_groups = historical_groups | parent_groups
    excluded = {key for key in groups if groups[key] in excluded_groups}
    independent = candidates - excluded
    size_counts = Counter(groups.values())
    remaining_groups = {groups[key] for key in independent}
    source_counts = {}
    for source in sorted({row["dataset_source"] for row in candidate_rows}):
        selected = [row for row in candidate_rows if row["dataset_source"] == source]
        keys = {row["report_hash"] for row in selected}
        available = keys & independent
        source_counts[source] = {
            "rows": len(selected), "unique_hashes": len(keys),
            "remaining_rows": sum(row["report_hash"] in independent for row in selected),
            "remaining_unique_hashes": len(available),
            "remaining_components": len({groups[key] for key in available}),
        }
    length_counts = {}
    for name, low, high in (("under_1000", 0, 999), ("1000_to_1499", 1000, 1499),
                            ("1500_to_1999", 1500, 1999), ("2000_to_4000", 2000, 4000),
                            ("over_4000", 4001, 10 ** 12)):
        keys = {key for key in independent if low <= len(normalize(texts[key])) <= high}
        length_counts[name] = {"remaining_unique_hashes": len(keys),
                              "remaining_components": len({groups[key] for key in keys})}
    pending = {
        "human_review_completed": False, "new_final_set_selected": False,
        "model_inference_performed": False, "patient_independence": "unknown",
        "fragment_parent_provenance": "Textual candidates only; no verified original parent IDs",
        "pretraining_exposure": "unknown", "limits": config["limits"],
        "historical_usage_unknown_count": manifest["summary"]["usage_status_counts"].get("unknown", 0),
        "grouping_scope": "Registered historical input hashes plus all local official validation source rows",
        "remaining_component_counts_are_not_additive_across_strata_or_sources": True,
    }
    result = {"version": config["version"], "config_sha256": config_sha,
              "groups": groups, "excluded_for_eval_hashes": sorted(excluded),
              "historical_hashes": sorted(historical), "pending_and_limits": pending}
    summary = {
        "version": config["version"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha, "source_files": dict(sorted(sources.items())),
        "historical": {"unique_hashes": len(historical),
                       "usage_status_counts": manifest["summary"]["usage_status_counts"],
                       "all_including_reserved_excluded": historical <= excluded},
        "grouping": {"unique_hashes": len(groups), "components": len(size_counts),
                     "components_with_multiple_hashes": sum(n > 1 for n in size_counts.values()),
                     "largest_component_hashes": max(size_counts.values()),
                     "edge_counts": dict(Counter(edge["type"] for edge in edges)),
                     "template_family_count": len(template_families),
                     "template_edges_are_component_spanning_stars_not_all_pairs": True,
                     "empty_5gram_sets": empty_sets, "unique_word_5grams": vocabulary_size,
                     "exact_join": join_counts},
        "v1_cross_split_text_family_conflicts": split_conflicts(groups, v1_splits, edges),
        "diagnostic_fragments": {
            "unique_fragments": len(fragments),
            "with_possible_containing_parent": sum(bool(v) for v in fragment_links.values()),
            "with_multiple_possible_parents": sum(len(v) > 1 for v in fragment_links.values()),
            "possible_parent_links": sum(map(len, fragment_links.values())),
            "unique_possible_parent_hashes": len(parent_hashes),
            "confirmed_parent_provenance_links": 0,
            "containment_links_join_components": False,
            "additional_candidate_hashes_excluded_via_parent_components":
                sum(groups[key] in parent_groups - historical_groups for key in candidates),
        },
        "validation_availability": {
            "source_rows": len(candidate_rows), "unique_source_hashes": len(candidates),
            "duplicate_source_rows": len(candidate_rows) - len(candidates),
            "exact_historical_hashes": len(candidates & historical),
            "exact_historical_rows": sum(row["report_hash"] in historical for row in candidate_rows),
            "additional_hashes_excluded_by_historical_text_family":
                sum(key not in historical and groups[key] in historical_groups for key in candidates),
            "excluded_hashes_total": len(candidates & excluded),
            "remaining_rows": sum(row["report_hash"] in independent for row in candidate_rows),
            "remaining_unique_hashes": len(independent),
            "remaining_components": len(remaining_groups),
            "by_dataset_source": source_counts, "by_normalized_character_length": length_counts,
            "final_test_selected": False,
        },
        "pending_and_limits": pending, "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    local = {"groups.json": result, "edges.json": {"edges": edges},
             "template_families.json": {"families": template_families},
             "fragment_parent_candidates.json": {"candidate_parents": fragment_links,
                 "relation": config["diagnostic_fragments"]["parent_relation"],
                 "confirmed_provenance": False, "unioned": False},
             "validation_source_inventory.json": {"rows": candidate_rows,
                 "remaining_hashes": sorted(independent), "final_test_selected": False}}
    return local, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.summary.exists():
        parser.error("Refusing to overwrite an existing audit")
    raw = args.config.read_bytes()
    config = json.loads(raw)
    local, summary = audit(args.root.resolve(), config, sha256(raw))
    summary["implementation_sha256"] = sha256(Path(__file__).read_bytes())
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in local.items():
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["local_artifacts_sha256"] = {path.name: sha256(path.read_bytes())
                                       for path in sorted(args.out.iterdir())}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": summary["elapsed_seconds"],
                      "historical_hashes": summary["historical"]["unique_hashes"],
                      "remaining_unique_hashes": summary["validation_availability"]["remaining_unique_hashes"],
                      "remaining_components": summary["validation_availability"]["remaining_components"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
