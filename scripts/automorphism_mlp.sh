cd ../src_synthetic/graph_separation || exit

python3 automorphism.py --seed 777 --hard --eval_sample_size 100

cd ../../scripts || exit
