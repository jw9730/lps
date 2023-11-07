cd ../src_synthetic/nbody || exit

python3 main.py --hard
python3 main.py --hard --test --test_sample_size 200

cd ../../scripts || exit
