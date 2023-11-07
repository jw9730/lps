cd ../src_synthetic/graph_separation || exit

python3 exp_iso.py --hard --backbone gin --noise_scale 0 --sample_size 1 --eval_sample_size 1

cd ../../scripts || exit
