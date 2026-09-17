"""Command line entry point."""

import argparse
from pathlib import Path

from .augmentation import augment
from .raw import inspect_macs, prepare_wide


def main() -> None:
    parser = argparse.ArgumentParser(prog="cgcr")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-raw", help="Count MAC addresses in a raw ZIP")
    inspect.add_argument("--archive", type=Path, required=True)
    inspect.add_argument("--report", type=Path, default=Path("outputs/mac_inventory.csv"))
    prepare = commands.add_parser("prepare", help="Build wide tables with a verified MAC-to-ID map")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--beacon-map", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    aug = commands.add_parser("augment", help="Synthesize minority room segments")
    aug.add_argument("--wide", type=Path, required=True)
    aug.add_argument("--output", type=Path, default=Path("data/synthetic.csv"))
    aug.add_argument("--repeats", type=int, default=6)
    aug.add_argument("--seed", type=int, default=42)
    fit = commands.add_parser("train", help="Train C0-C3 and save a model bundle")
    fit.add_argument("--labeled", type=Path, required=True)
    fit.add_argument("--synthetic", type=Path, required=True)
    fit.add_argument("--unlabeled", type=Path, required=True)
    fit.add_argument("--model", type=Path, default=Path("models/cgcr.joblib"))
    infer = commands.add_parser("predict", help="Predict test rows with a saved bundle")
    infer.add_argument("--model", type=Path, default=Path("models/cgcr.joblib"))
    infer.add_argument("--test", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.command == "inspect-raw":
        print(f"Found {inspect_macs(args.archive, args.report)} MAC addresses; saved {args.report}")
    elif args.command == "prepare":
        print(prepare_wide(args.archive, args.beacon_map, args.output_dir))
    elif args.command == "augment":
        result = augment(args.wide, args.output, args.repeats, args.seed)
        print(f"Saved {len(result)} rows to {args.output}")
    elif args.command == "train":
        from .pipeline import train

        result = train(args.labeled, args.synthetic, args.unlabeled, args.model)
        for item in result["summary"]:
            print(item)
        print(f"Saved {args.model}")
    else:
        from .pipeline import predict

        for path in predict(args.model, args.test, args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
