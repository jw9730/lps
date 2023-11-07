cd ../src_synthetic/nbody || exit

python3 main.py --hard --interface unif
python3 main.py --hard --interface unif --test --test_sample_size 200

cd ../../scripts || exit
