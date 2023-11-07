cd ../src_synthetic/graph_separation || exit

python3 exp_classify.py --hard --backbone gin --num_epochs 1000

cd ../../scripts || exit
