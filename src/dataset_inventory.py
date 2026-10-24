"""Inventory TrackML event files without loading physics features.

The inventory records CSV data-row counts for hits, cells, particles, and
truth. It accepts either an extracted dataset directory or a ZIP archive, so
large raw datasets do not need to be copied into the repository.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Callable, TextIO


DATA_PARTS = ("hits", "cells", "particles", "truth")
_EVENT_FILE = re.compile(r"^event(\d{9})-(hits|cells|particles|truth)\.csv$")


def _data_row_count(stream: TextIO) -> int:
    """Count CSV records excluding the header, respecting quoted newlines."""
    rows = csv.reader(stream)
    next(rows, None)
    return sum(1 for _ in rows)


def inventory_dataset(source: str | Path) -> list[dict[str, int | str]]:
    """Return a sorted event inventory from a directory or ZIP archive.

    ``event_id`` is the numeric ID used by :func:`src.data_loader.load_event`.
    Missing parts have a row count of zero and mark the event incomplete.
    Duplicate files for the same event and part are rejected as ambiguous.
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source_path}")

    entries: dict[int, dict[str, Path | str]] = {}

    def add_entry(name: str, locator: Path | str) -> None:
        match = _EVENT_FILE.fullmatch(Path(name).name)
        if not match:
            return
        event_id, part = int(match.group(1)), match.group(2)
        event_parts = entries.setdefault(event_id, {})
        if part in event_parts:
            raise ValueError(f"Duplicate {part} file for event {event_id}")
        event_parts[part] = locator

    if source_path.is_dir():
        for path in source_path.rglob("event*-*.csv"):
            add_entry(path.name, path)
        def open_part(locator: Path | str) -> TextIO:
            return Path(locator).open("r", encoding="utf-8-sig", newline="")
    elif source_path.is_file() and zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as archive:
            for member in archive.namelist():
                if not member.endswith("/"):
                    add_entry(member, member)

            def open_part(locator: Path | str) -> TextIO:
                return io.TextIOWrapper(
                    archive.open(str(locator), "r"), encoding="utf-8-sig", newline=""
                )

            return _build_inventory(entries, open_part, source_path)
    else:
        raise ValueError(f"Dataset source must be a directory or ZIP archive: {source_path}")

    return _build_inventory(entries, open_part, source_path)


def _build_inventory(
    entries: dict[int, dict[str, Path | str]],
    open_part: Callable[[Path | str], TextIO],
    source_path: Path,
) -> list[dict[str, int | str]]:
    """Count each discovered part and assemble one record per event."""
    inventory: list[dict[str, int | str]] = []
    for event_id in sorted(entries):
        parts = entries[event_id]
        row: dict[str, int | str] = {"event_id": event_id}
        for part in DATA_PARTS:
            locator = parts.get(part)
            if locator is None:
                row[f"{part}_rows"] = 0
            else:
                with open_part(locator) as stream:
                    row[f"{part}_rows"] = _data_row_count(stream)
        row["completeness_status"] = (
            "complete" if all(part in parts for part in DATA_PARTS) else "incomplete"
        )
        inventory.append(row)
    if not inventory:
        raise ValueError(f"No TrackML event CSV files found in {source_path}")
    return inventory


def write_inventory(inventory: list[dict[str, int | str]], output_path: str | Path) -> Path:
    """Write inventory records as a deterministic CSV and return its path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", *(f"{part}_rows" for part in DATA_PARTS), "completeness_status"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)
    return output
