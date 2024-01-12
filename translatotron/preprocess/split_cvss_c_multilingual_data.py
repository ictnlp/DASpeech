import os
import argparse
import pandas as pd

from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
)

langs = ["fr", "de", "es", "ca", "it", "ru", "zh-CN", "pt", "fa", "et", "mn", "nl", "tr", "ar", "sv-SE", "lv", "sl", "ta", "ja", "id", "cy"]

splits = ["train", "dev", "test"]


def process(args):
    data_root = Path(args.data_root)
    multilingual_root = data_root / "x-en"
    multilingual_subdir = multilingual_root / args.subdir_name
    # main tsv
    for split in splits:
        multilingual_df = load_df_from_tsv(multilingual_subdir / f"{split}.tsv")
        for lang in langs:
            lang_root = data_root / f"{lang}-en"
            lang_subdir = lang_root / args.subdir_name
            lang_subdir.mkdir(exist_ok=True)
            lang_df = multilingual_df[multilingual_df.id.str.contains(lang)]
            save_df_to_tsv(lang_df, lang_subdir / f"{split}.tsv")
    # multitask tsv
    for task in args.multitask_name.split(","):
        if task == "":
            return
        multilingual_taskdir = multilingual_root / task
        for split in splits:
            multilingual_df = load_df_from_tsv(multilingual_taskdir / f"{split}.tsv")
            for lang in langs:
                lang_root = data_root / f"{lang}-en"
                lang_taskdir = lang_root / task
                lang_taskdir.mkdir(exist_ok=True)
                lang_df = multilingual_df[multilingual_df.id.str.contains(lang)]
                save_df_to_tsv(lang_df, lang_taskdir / f"{split}.tsv")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--subdir-name", type=str, required=True)
    parser.add_argument("--multitask-name", default="")
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()