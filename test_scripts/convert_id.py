import os
import tqdm
import shutil
import argparse


from examples.speech_to_text.data_utils import load_df_from_tsv

def process(args):
    df = load_df_from_tsv(args.input_tsv)
    data = list(df.T.to_dict().values())
    for idx, item in tqdm.tqdm(enumerate(data)):
        old_path = os.path.join(args.audio_dir, f"{item['id']}_generated_e2e.wav")
        new_path = os.path.join(args.audio_dir, f"{idx}_pred.wav")
        shutil.copy(old_path, new_path)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv")
    parser.add_argument("--audio-dir")
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()
