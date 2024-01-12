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
    s2tt_tsv_dir = Path(args.s2tt_tsv_dir)
    tts_tsv_dir = Path(args.tts_tsv_dir)
    s2pt_tsv_dir = Path(args.s2pt_tsv_dir)
    s2pt_tsv_dir.mkdir(exist_ok=True)

    for split in ["train", "dev", "test"]:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        s2tt_df = load_df_from_tsv(s2tt_tsv_dir / f"{split}.tsv")
        s2tt_data = list(s2tt_df.T.to_dict().values())
        tts_df = load_df_from_tsv(tts_tsv_dir / f"{split}.tsv")
        tts_data = list(tts_df.T.to_dict().values())
        tts_data = {x["id"] : x for x in tts_data}
        for s2tt_item in tqdm.tqdm(s2tt_data):
            if f"{s2tt_item['id']}" in tts_data:
                manifest["id"].append(s2tt_item["id"])
                manifest["audio"].append(s2tt_item["audio"])
                manifest["n_frames"].append(s2tt_item["n_frames"])
                manifest["tgt_text"].append(tts_data[f"{s2tt_item['id']}"]["tgt_text"])
                manifest["speaker"].append("None")
        df = pd.DataFrame.from_dict(manifest)
        save_df_to_tsv(df, s2pt_tsv_dir / f"{split}.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2tt-tsv-dir")
    parser.add_argument("--tts-tsv-dir")
    parser.add_argument("--s2pt-tsv-dir")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()