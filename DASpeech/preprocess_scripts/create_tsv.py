import argparse
import pandas as pd

from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
)

SPLITS = ["train", "dev", "test"]

def process(args):
    s2t_root = Path(args.s2t_dir)
    tts_root = Path(args.tts_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(exist_ok=True)

    for split in SPLITS:
        s2t_df = load_df_from_tsv(s2t_root / f"{split}.tsv")
        tts_df = load_df_from_tsv(tts_root / f"{split}.tsv")
        s2t_df = s2t_df.rename(columns={"audio": "src_audio", "n_frames": "src_n_frames"})
        tts_df = tts_df.rename(columns={"audio": "tgt_audio", "n_frames": "tgt_n_frames"})
        s2t_df = s2t_df.drop(columns=["speaker"])
        tts_df = tts_df.drop(columns=["speaker", "src_text"])
        out_df = pd.merge(s2t_df, tts_df, how="inner", on=["id", "tgt_text"])
        assert len(out_df) == len(s2t_df) == len(tts_df)
        save_df_to_tsv(out_df, output_root / f"{split}.tsv")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2t-dir", type=str, required=True)
    parser.add_argument("--tts-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()