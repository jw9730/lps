cd ../src_synthetic/nbody || exit

python3 main.py --hard --backbone gnn --symmetry O3
python3 main.py --hard --backbone gnn --symmetry O3 --test

cd ../../scripts || exit
