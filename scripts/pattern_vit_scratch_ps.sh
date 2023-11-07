cd .. || exit

python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 10
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 10 --test_seed 1
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 10 --test_seed 2
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 10 --test_seed 3
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 10 --test_seed 4
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 10 --test_seed 5
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 1
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 1 --test_seed 1
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 1 --test_seed 2
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 1 --test_seed 3
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 1 --test_seed 4
python3 main.py --config configs/node_classification_pattern/vit_scratch_ps.yaml --test_mode --test_sample_size 1 --test_seed 5

cd scripts || exit
