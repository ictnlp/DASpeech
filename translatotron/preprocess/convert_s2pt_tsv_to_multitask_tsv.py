import tqdm
import argparse
import pandas as pd

from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    s2pt_tsv_dir = Path(args.s2pt_tsv_dir)
    multitask_dir = Path(args.multitask_dir)
    multitask_dir.mkdir(exist_ok=True)

    for split in ["train", "dev", "test"]:
        s2pt_df = load_df_from_tsv(s2pt_tsv_dir/ f"{split}.tsv")
        s2pt_df = s2pt_df.drop(columns=["audio", "n_frames", "speaker"])
        save_df_to_tsv(s2pt_df, multitask_dir / f"{split}.tsv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2pt-tsv-dir")
    parser.add_argument("--multitask-dir")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()