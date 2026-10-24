"""Deterministic whole-event split manifest for TrackML events 1000–1099."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


SPLITS = ("DEVELOPMENT", "VALIDATION", "FINAL_TEST")
MANIFEST_FIELDS = ("event_id", "split")


def expected_manifest() -> list[dict[str, str]]:
    """Return the fixed, ordered event-to-split mapping for M7.2."""
    rows: list[dict[str, str]] = []
    for event_number in range(1000, 1100):
        if event_number < 1060:
            split = "DEVELOPMENT"
        elif event_number < 1080:
            split = "VALIDATION"
        else:
            split = "FINAL_TEST"
        rows.append({"event_id": f"event{event_number:09d}", "split": split})
    return rows


def write_manifest(path: str | Path) -> Path:
    """Write the fixed manifest in event-ID order and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(expected_manifest())
    return output_path


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read event/split records from a CSV manifest."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(MANIFEST_FIELDS).issubset(reader.fieldnames):
            raise ValueError("Manifest must contain event_id and split columns")
        return [
            {"event_id": (row["event_id"] or "").strip(), "split": (row["split"] or "").strip()}
            for row in reader
        ]


def validate_manifest(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    """Validate IDs, uniqueness, exact split membership, and split sizes.

    Raises ``ValueError`` with all detected issues. Returns counts by split
    when the manifest is valid.
    """
    normalized: list[dict[str, str]] = []
    for row in rows:
        if "event_id" not in row or "split" not in row:
            raise ValueError("Every manifest row must contain event_id and split")
        normalized.append(
            {"event_id": str(row["event_id"]).strip(), "split": str(row["split"]).strip()}
        )

    issues: list[str] = []
    if len(normalized) != 100:
        issues.append(f"expected exactly 100 rows, found {len(normalized)}")

    ids = [row["event_id"] for row in normalized]
    duplicates = sorted(event_id for event_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        issues.append(f"duplicate event IDs: {', '.join(duplicates)}")

    expected_rows = expected_manifest()
    expected_by_id = {row["event_id"]: row["split"] for row in expected_rows}
    actual_ids = set(ids)
    expected_ids = set(expected_by_id)
    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)
    if missing_ids or unexpected_ids:
        if missing_ids:
            issues.append(f"missing expected event IDs: {', '.join(missing_ids)}")
        if unexpected_ids:
            issues.append(f"unexpected event IDs: {', '.join(unexpected_ids)}")

    split_memberships: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in normalized:
        event_id, split = row["event_id"], row["split"]
        split_memberships[event_id].add(split)
        split_counts[split] += 1
        if split not in SPLITS:
            issues.append(f"invalid split {split!r} for {event_id}")
        if event_id in expected_by_id and split != expected_by_id[event_id]:
            issues.append(
                f"{event_id} belongs in {expected_by_id[event_id]}, found {split}"
            )

    multi_split_ids = sorted(
        event_id for event_id, assigned_splits in split_memberships.items()
        if len(assigned_splits) > 1
    )
    if multi_split_ids:
        issues.append(f"events appear in multiple splits: {', '.join(multi_split_ids)}")

    required_counts = {"DEVELOPMENT": 60, "VALIDATION": 20, "FINAL_TEST": 20}
    for split, required in required_counts.items():
        found = split_counts[split]
        if found != required:
            issues.append(f"expected {required} {split} events, found {found}")

    if issues:
        raise ValueError("Invalid event split manifest: " + "; ".join(issues))
    return dict(required_counts)
