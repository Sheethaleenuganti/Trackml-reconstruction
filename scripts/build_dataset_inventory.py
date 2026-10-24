"""Create a row-count inventory of TrackML events."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset_inventory import inventory_dataset, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="Extracted dataset directory or ZIP archive (for example train_sample.zip)",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/dataset_inventory.csv",
        help="Output CSV path (default: %(default)s)",
    )
    args = parser.parse_args()

    rows = inventory_dataset(args.source)
    output = write_inventory(rows, args.output)
    complete = sum(row["completeness_status"] == "complete" for row in rows)
    print(f"Events inventoried: {len(rows)} ({complete} complete)")
    print(f"Inventory written to: {output.resolve()}")


if __name__ == "__main__":
    main()
