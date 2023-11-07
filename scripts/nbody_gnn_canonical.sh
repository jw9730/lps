cd ../src_synthetic/nbody || exit

python3 main.py --hard --backbone gnn --symmetry O3 --noise_scale 0 --sample_size 1 --eval_sample_size 1 --test_sample_size 1 --test_n_trials 1
python3 main.py --hard --backbone gnn --symmetry O3 --noise_scale 0 --sample_size 1 --eval_sample_size 1 --test_sample_size 1 --test_n_trials 1 --test

cd ../../scripts || exit
