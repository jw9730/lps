cd ../src_synthetic/nbody || exit

python3 main.py --hard --backbone gnn --symmetry O3 --interface unif
python3 main.py --hard --backbone gnn --symmetry O3 --interface unif --test

cd ../../scripts || exit
