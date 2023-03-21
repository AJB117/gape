seed0=41
seed1=95
seed2=12
seed3=35

# no PE
log_file=CSL_graph_classification_no_pe.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-baseline-trials.json --job_num 1 --pos_enc_dim 1 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# LAPE
log_file=CSL_graph_classification_lape.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-trials.json --job_num 2 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# RW
log_file=CSL_graph_classification_rw.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-prwpe-trials.json --job_num 3 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SA
log_file=CSL_graph_classification_sa.log
python3 main_CSL_graph_classification.py --config tests/test-configs/SAGraphTransformer_CSL_CSL_b5-300k-sa-trials.json --job_num 4 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SPD+C
log_file=CSL_graph_classification_spdc.log
python3 main_CSL_graph_classification.py --config tests/test-configs/PseudoGraphormer_CSL_CSL_b5-spd-300k-trials.json --job_num 5 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE
log_file=CSL_graph_classification_gape.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-rand-scale002-trials.json --job_num 6 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*
log_file=CSL_graph_classification_gape_stoch.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-rand-stoch-trials.json --job_num 7 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**
log_file=CSL_graph_classification_gape_stoch_softinit.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-rand-stoch-softinit-trials.json --job_num 8 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*_20
log_file=CSL_graph_classification_gape_stoch_20.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-rand-stoch-initials20-trials.json --job_num 9 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_20
log_file=CSL_graph_classification_gape_stoch_softinit_20.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-rand-stoch-softinit-initials20-trials.json --job_num 10 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_max
log_file=CSL_graph_classification_gape_stoch_softinit_max.log
python3 main_CSL_graph_classification.py --config tests/test-configs/GraphTransformer_CSL_CSL_b5-300k-rand-stoch-softinit-initials100-topn-trials.json --job_num 11 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;