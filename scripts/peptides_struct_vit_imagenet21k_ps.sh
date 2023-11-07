cd .. || exit

python3 main.py --config configs/graph_regression_peptides_struct/vit_imagenet21k_ps.yaml
python3 main.py --config configs/graph_regression_peptides_struct/vit_imagenet21k_ps.yaml --test_mode --test_sample_size 10

cd scripts || exit
