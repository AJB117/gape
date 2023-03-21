seed0=41
seed1=95
seed2=12
seed3=35

# no PE
log_file=ZINC_graph_regression_no_pe.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-trials.json --job_num 1 --pos_enc_dim 1 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# LAPE
log_file=ZINC_graph_regression_lape.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-8-trials.json --job_num 2 --pos_enc_dim 8 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# RW
log_file=ZINC_graph_regression_rw.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-prwpe-noedge-500k-trials.json --job_num 3 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SA
log_file=ZINC_graph_regression_sa.log
python3 main_molecules_graph_regression.py --config tests/test-configs/SAGraphTransformer_molecules_ZINC_b128-noedge-500k-trials.json --job_num 4 --pos_enc_dim 10 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SPD+C
log_file=ZINC_graph_regression_spdc.log
python3 main_molecules_graph_regression.py --config tests/test-configs/PseudoGraphormer_molecules_ZINC_b128-500k-trials.json --job_num 5 --pos_enc_dim 1 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE
log_file=ZINC_graph_regression_gape.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-32-scale50-trials.json --job_num 6 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*
log_file=ZINC_graph_regression_gape_stoch.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-32-stoch-trials.json --job_num 7 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**
log_file=ZINC_graph_regression_gape_stoch_softinit.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-32-stoch-softinit-trials.json --job_num 8 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*_20
log_file=ZINC_graph_regression_gape_stoch_20.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-32-stoch-initials20-trials.json --job_num 9 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_20
log_file=ZINC_graph_regression_gape_stoch_softinit_20.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-32-stoch-softinit-initials20-trials.json --job_num 10 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_max
log_file=ZINC_graph_regression_gape_stoch_softinit_max.log
python3 main_molecules_graph_regression.py --config tests/test-configs/GraphTransformer_molecules_ZINC_b128-bnorm-alt-noedge-500k-32-stoch-softinit-initials100-topn-trials.json --job_num 11 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;
