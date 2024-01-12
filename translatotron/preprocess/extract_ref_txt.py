import os
import argparse
from examples.speech_to_text.data_utils import load_df_from_tsv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=str, required=True)
    parser.add_argument("--output-txt", type=str, required=True)
    args = parser.parse_args()
    df = load_df_from_tsv(args.input_tsv)
    data = list(df.T.to_dict().values())
    with open(args.output_txt, "w") as f:
        for item in data:
            f.write(item["tgt_text"].lower() + "\n")


if __name__ == "__main__":
    main()