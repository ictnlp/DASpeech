DATA_ROOT=$1

ROOT=~/DASpeech
KM_MODEL_PATH=$ROOT/translatotron/preprocess/s2ut/mhubert.km1000.layer11.pt
CKPT_PATH=$ROOT/checkpoints/mhubert_base_vp_en_es_fr_it3.pt

python $ROOT/translatotron/preprocess/s2ut/create_manifest.py --data-root $DATA_ROOT

for split in train dev test
do
    python $ROOT/translatotron/preprocess/s2ut/quantize_with_kmeans.py \
        --feature_type hubert \
        --kmeans_model_path $KM_MODEL_PATH \
        --acoustic_model_path $CKPT_PATH \
        --layer 11 \
        --manifest_path $DATA_ROOT/$split.txt \
        --out_quantized_file_path $DATA_ROOT/$split.km1000
done