import argparse
import logging

from src.pipelines.generate import run_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True,
                        help="Путь к train_val_videodatainfo.json (MSR-VTT)")
    parser.add_argument("--output", required=True,
                        help="Путь к выходному JSONL")
    parser.add_argument("--split", default="validate",
                        choices=["train", "validate", "test"])
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_generation(
        annotations_path=args.annotations,
        output_path=args.output,
        split=args.split,
        classifier_threshold=args.threshold,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    main()
