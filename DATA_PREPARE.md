# Data Preparation

This is a tutorial about data processing. 

> Note: This is a version from our development process, and there might be some unnecessary steps. We release it because many people need it. We will soon release a more concise version.

## 1. Extract Discrete Units (Unnecessary for DASpeech but used in S2UT and UnitY)

```
sh translatotron/preprocess/s2ut/run_mhubert.sh data/cvss-c/fr-en
```

## 2. Prepare speech-to-unit tsv

> Note: This TSV is prepared for subsequent speech-to-text/speech TSV processing.

```
python translatotron/preprocess/prep_cvss_c_multilingual_data.py \
    --covost-data-root data/covost2/ \
    --cvss-data-root data/cvss-c/ \
    --output-root data/cvss-c/x-en/ \
    --target-type unit --unit-type km1000 --reduce-unit \
    --vocoder-checkpoint vocoder/mhubert_lyr11_km1000_en/g_00500000 \
    --vocoder-cfg vocoder/mhubert_lyr11_km1000_en/config.json 
```

The obtained TSV above contains data in all languages (X-En). To conduct monolingual experiments, run the following script to split it into folders for each language:

```
python translatotron/preprocess/split_cvss_c_multilingual_data.py \
    --data-root data/cvss-c \
    --subdir-name fbank2unit
```

> Note: config.yaml may need to be copied to each sub-directory manually.

## 3. Convert speech-to-unit tsv into speech-to-text tsv

```
python translatotron/preprocess/convert_s2st_tsv_to_s2tt_tsv.py \
    --s2st-tsv-dir data/cvss-c/x-en/fbank2unit/ \
    --s2tt-tsv-dir data/cvss-c/x-en/fbank2text/

python translatotron/preprocess/split_cvss_c_multilingual_data.py \
    --data-root data/cvss-c \
    --subdir-name fbank2text
```

## 4. Forced Alignment for the Target Speech using [MFA](https://mfa-models.readthedocs.io/en/latest/index.html)

First, you need to organize the data into the format required by MFA:

```
python translatotron/preprocess/reorganize_mfa_format.py \
    --data-root data/cvss-c/ \
    --lang fr
```

```
mfa align data/cvss-c/fr-en/mfa/ \
    english_us_arpa english_us_arpa \
    data/cvss-c/fr-en/mfa_align \
    -j 8 -v --clean --single_speaker
```

Alignment results will be output to the `data/cvss-c/fr-en/mfa_align` folder.

## 5. Prepare the text-to-speech tsv

```
python translatotron/preprocess/prep_cvss_c_tts_data.py \
    --audio-manifest-root data/cvss-c/fr-en/ \
    --output-root data/cvss-c/fr-en/tts \
    --textgrid-dir data/cvss-c/fr-en/mfa_align/speaker/
```

Perform an additional step to remove ".mp3" from the IDs:

```
sed -i "s/.mp3//" data/cvss-c/fr-en/tts/train.tsv
sed -i "s/.mp3//" data/cvss-c/fr-en/tts/dev.tsv
sed -i "s/.mp3//" data/cvss-c/fr-en/tts/test.tsv
```

## 6. Prepare the speech-to-phoneme tsv

```
python translatotron/preprocess/convert_s2tt_tsv_to_s2pt_tsv.py \
    --s2tt-tsv-dir data/cvss-c/fr-en/fbank2text/ \
    --tts-tsv-dir data/cvss-c/fr-en/tts \
    --s2pt-tsv-dir data/cvss-c/fr-en/fbank2phone
```

Manually copy the vocabulary file `tts/vocab.txt` to the `fbank2phone/` directory, and follow the instructions in the README to write the `config.yaml`.


## 7. Prepare the multitask (tgt_phoneme) tsv

```
python translatotron/preprocess/convert_s2pt_tsv_to_multitask_tsv.py \
    --s2pt-tsv-dir data/cvss-c/fr-en/fbank2phone \
    --multitask-dir data/cvss-c/fr-en/tgt_phoneme
```

## 8. Merge the speech-to-phoneme and TTS tsv to obtain the final S2ST tsv

```
python DASpeech/preprocess_scripts/create_tsv.py \
    --s2t-dir data/cvss-c/fr-en/fbank2phone \
    --tts-dir data/cvss-c/fr-en/tts \
    --output-dir data/cvss-c/fr-en/nat_s2s
```

## 9. Prepare the tsv for evaluation

Due to the possibility of some data being missing during the above processing steps (due to alignment or pitch/energy extraction failures), to ensure the completeness of testing during evaluation, prepare a complete test set TSV:

```
python DASpeech/preprocess_scripts/create_test_tsv.py \
    --s2t-dir data/cvss-c/fr-en/fbank2unit \
    --output-dir data/cvss-c/fr-en/nat_s2s
```
