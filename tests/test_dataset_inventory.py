"""Focused tests for directory and ZIP dataset inventory."""
import csv
import zipfile

from src.dataset_inventory import inventory_dataset, write_inventory


def _write_event_part(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "value"])
        writer.writerows(rows)


def test_directory_inventory_counts_data_rows_and_marks_incomplete(tmp_path):
    event_dir = tmp_path / "train_100_events"
    event_dir.mkdir()
    for part, count in (("hits", 2), ("cells", 3), ("particles", 1)):
        _write_event_part(
            event_dir / f"event000001000-{part}.csv",
            [[i, f"value-{i}"] for i in range(count)],
        )

    rows = inventory_dataset(event_dir)

    assert rows == [
        {
            "event_id": 1000,
            "hits_rows": 2,
            "cells_rows": 3,
            "particles_rows": 1,
            "truth_rows": 0,
            "completeness_status": "incomplete",
        }
    ]


def test_zip_inventory_counts_complete_event_and_writes_csv(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for part, count in (("hits", 2), ("cells", 3), ("particles", 1), ("truth", 2)):
            contents = "id,value\n" + "".join(f"{i},v{i}\n" for i in range(count))
            archive.writestr(f"train_100_events/event000001001-{part}.csv", contents)

    rows = inventory_dataset(archive_path)
    output = write_inventory(rows, tmp_path / "results" / "inventory.csv")

    assert rows == [
        {
            "event_id": 1001,
            "hits_rows": 2,
            "cells_rows": 3,
            "particles_rows": 1,
            "truth_rows": 2,
            "completeness_status": "complete",
        }
    ]
    with output.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["event_id"] == "1001"
    assert saved[0]["completeness_status"] == "complete"
