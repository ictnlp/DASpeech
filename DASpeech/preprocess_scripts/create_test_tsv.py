import argparse
import pandas as pd

from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
)

def process(args):
    s2t_root = Path(args.s2t_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(exist_ok=True)

    for split in ["dev", "test"]:
        s2t_df = load_df_from_tsv(s2t_root / f"{split}.tsv")
        s2t_df = s2t_df.drop(columns=["src_text", "tgt_text", "tgt_audio", "tgt_n_frames"])
        s2t_df["id"] = s2t_df["id"] + ".mp3"
        save_df_to_tsv(s2t_df, output_root / f"{split}.full.tsv")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2t-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()