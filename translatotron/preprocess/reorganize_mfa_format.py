import shutil
import argparse
from pathlib import Path

SPLITS = ["train", "dev", "test"]


def process(args):
    data_root = Path(args.data_root) / f"{args.lang}-en"
    mfa_dir = data_root / "mfa"
    mfa_dir.mkdir(exist_ok=True)
    output_dir = mfa_dir / "speaker"
    output_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        with open(data_root / f"{split}.tsv") as f:
            data = f.read().splitlines()
        for item in data:
            idx, text = item.split("\t")
            shutil.copy(data_root / split / f"{idx}.wav", output_dir)
            with open(output_dir / f"{idx}.lab", "w") as f:
                f.write(text.upper())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()