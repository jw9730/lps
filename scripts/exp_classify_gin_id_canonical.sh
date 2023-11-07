cd ../src_synthetic/graph_separation || exit

python3 exp_classify.py --hard --backbone gin --noise_scale 0 --sample_size 1 --eval_sample_size 1 --num_epochs 1000

cd ../../scripts || exit
