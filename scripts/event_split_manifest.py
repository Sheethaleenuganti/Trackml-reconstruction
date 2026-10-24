"""Create or validate the deterministic M7.2 event split manifest."""
from __future__ import annotations

import argparse

from src.event_split_manifest import read_manifest, validate_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="write the fixed 100-event manifest")
    create_parser.add_argument(
        "path", nargs="?", default="experiments/results/event_split_manifest.csv"
    )

    validate_parser = subparsers.add_parser("validate", help="validate a manifest CSV")
    validate_parser.add_argument(
        "path", nargs="?", default="experiments/results/event_split_manifest.csv"
    )

    args = parser.parse_args()
    if args.command == "create":
        path = write_manifest(args.path)
        print(f"Wrote 100 event assignments to {path.resolve()}")
    else:
        counts = validate_manifest(read_manifest(args.path))
        print(
            "Manifest valid: "
            f"{counts['DEVELOPMENT']} development, "
            f"{counts['VALIDATION']} validation, "
            f"{counts['FINAL_TEST']} final-test events"
        )


if __name__ == "__main__":
    main()
