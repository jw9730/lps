cd .. || exit

python3 main.py --config configs/node_classification_pattern/vit_scratch_canonical.yaml
python3 main.py --config configs/node_classification_pattern/vit_scratch_canonical.yaml --test_mode --test_batch_size 128 --test_sample_size 1

cd scripts || exit
