cd .. || exit

python3 main.py --config configs/graph_classification_peptides_func/vit_imagenet21k_ps.yaml
python3 main.py --config configs/graph_classification_peptides_func/vit_imagenet21k_ps.yaml --test_mode --test_batch_size 8 --test_sample_size 100
python3 main.py --config configs/graph_classification_peptides_func/vit_imagenet21k_ps.yaml --test_mode --test_batch_size 8 --test_sample_size 100 --test_seed 1
python3 main.py --config configs/graph_classification_peptides_func/vit_imagenet21k_ps.yaml --test_mode --test_batch_size 8 --test_sample_size 100 --test_seed 2

cd scripts || exit
