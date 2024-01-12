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
    s2st_tsv_dir = Path(args.s2st_tsv_dir)
    s2tt_tsv_dir = Path(args.s2tt_tsv_dir)
    s2tt_tsv_dir.mkdir(exist_ok=True)

    for split in ["train", "dev", "test"]:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        df = load_df_from_tsv(s2st_tsv_dir / f"{split}.tsv")
        data = list(df.T.to_dict().values())
        for item in tqdm.tqdm(data):
            manifest["id"].append(item["id"])
            manifest["audio"].append(item["src_audio"])
            manifest["n_frames"].append(item["src_n_frames"])
            manifest["tgt_text"].append(item["tgt_text"])
            manifest["speaker"].append("None")
        df = pd.DataFrame.from_dict(manifest)
        save_df_to_tsv(df, s2tt_tsv_dir / f"{split}.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2st-tsv-dir")
    parser.add_argument("--s2tt-tsv-dir")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()