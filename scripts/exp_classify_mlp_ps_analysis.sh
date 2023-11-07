cd ../src_synthetic/graph_separation || exit

# run training
# probabilistic symmetrization in comparison to group averaging
python3 exp_classify.py --hard --skip_if_run_exists
python3 exp_classify.py --hard --interface unif --skip_if_run_exists
# probabilistic symmetrization with different sample sizes
python3 exp_classify.py --hard --sample_size 1 --eval_sample_size 10 --skip_if_run_exists
python3 exp_classify.py --hard --sample_size 2 --eval_sample_size 10 --skip_if_run_exists
python3 exp_classify.py --hard --sample_size 5 --eval_sample_size 10 --skip_if_run_exists
python3 exp_classify.py --hard --sample_size 10 --eval_sample_size 10 --skip_if_run_exists
python3 exp_classify.py --hard --sample_size 20 --eval_sample_size 10 --skip_if_run_exists
python3 exp_classify.py --hard --sample_size 50 --eval_sample_size 10 --skip_if_run_exists

# run analysis
# probabilistic symmetrization in comparison to group averaging
python3 exp_classify_analysis.py --hard --all_epochs
python3 exp_classify_analysis.py --hard --interface unif --all_epochs
# probabilistic symmetrization with different sample sizes
python3 exp_classify_analysis.py --hard --sample_size 1 --eval_sample_size 10
python3 exp_classify_analysis.py --hard --sample_size 2 --eval_sample_size 10
python3 exp_classify_analysis.py --hard --sample_size 5 --eval_sample_size 10
python3 exp_classify_analysis.py --hard --sample_size 10 --eval_sample_size 10
python3 exp_classify_analysis.py --hard --sample_size 20 --eval_sample_size 10
python3 exp_classify_analysis.py --hard --sample_size 50 --eval_sample_size 10
# compile results
python3 exp_classify_analysis.py --compile_results

cd ../../scripts || exit
