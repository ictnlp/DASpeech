exp=$1
beta=$2

checkpoint_dir=checkpoints/$exp
output_dir=results/${exp}_viterbi_$beta

mkdir -p $output_dir

python fairseq/scripts/average_checkpoints.py \
    --inputs $checkpoint_dir/ \
    --num-update-checkpoints 5 \
    --checkpoint-upper-bound 50000 \
    --output $checkpoint_dir/average_last_5_upper_50000.pt

python DASpeech/generator/generate_features.py \
  data/cvss-c/fr-en/s2s \
  --user-dir DASpeech \
  --config-yaml config.yaml --gen-subset test --task nat_speech_to_speech \
  --path $checkpoint_dir/average_last_5_upper_50000.pt --max-tokens 40000 --spec-bwd-max-iter 32 \
  --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
  --model-overrides "{\"decode_strategy\":\"jointviterbi\",\"decode_viterbibeta\":$beta,\"decode_beta\":1}" \
  --required-batch-size-multiple 1 \
  --results-path $output_dir \
  --generator-type nat_s2s

python hifi-gan/inference_e2e.py \
    --input_mels_dir $output_dir/feat \
    --output_dir $output_dir/wav \
    --checkpoint_file hifi-gan/VCTK_V1/generator_v1

python test_scripts/convert_id.py \
    --input-tsv data/cvss-c/fr-en/s2s/test.tsv \
    --audio-dir $output_dir/wav

cd asr_bleu/
python compute_asr_bleu.py \
  --lang en \
  --audio_dirpath ../$output_dir/wav \
  --reference_path ../data/cvss-c/fr-en/test.txt \
  --reference_format txt