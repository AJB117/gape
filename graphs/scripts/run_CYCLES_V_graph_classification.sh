seed0=41
seed1=95
seed2=12
seed3=35

# no PE
log_file=CYCLES_V_graph_classification_no_pe.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-nope-vary-trials.json --job_num 1 --pos_enc_dim 1 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# LAPE
log_file=CYCLES_V_graph_classification_lape.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-vary-trials.json --job_num 2 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# RW
log_file=CYCLES_V_graph_classification_rw.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-prwpe-vary-trials.json --job_num 3 --pos_enc_dim 9 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SA
log_file=CYCLES_V_graph_classification_sa.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/SAGraphTransformer_CYCLES_CYCLES_b25-noedge-500k-lpe1-lpedim8-vary-trials.json --job_num 4 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SPD+C
log_file=CYCLES_V_graph_classification_spdc.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/PseudoGraphormer_CYCLES_CYCLES_b25-500k-vary-trials.json --job_num 5 --pos_enc_dim 1 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE
log_file=CYCLES_V_graph_classification_gape.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials-32-vary-scale002.json --job_num 6 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*
log_file=CYCLES_V_graph_classification_gape_stoch.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials-32-stoch-vary.json --job_num 7 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**
log_file=CYCLES_V_graph_classification_gape_stoch_softinit.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials-32-stoch-vary-softinit.json --job_num 8 --pos_enc_dim 32 --log_file $fname --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*_20
log_file=CYCLES_V_graph_classification_gape_stoch_20.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials-32-stoch-vary-initials20.json --job_num 9 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_20
log_file=CYCLES_V_graph_classification_gape_stoch_softinit_20.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials-32-stoch-vary-softinit-initials20.json --job_num 10 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_max
log_file=CYCLES_V_graph_classification_gape_stoch_softinit_max.log
python3 main_CYCLES_graph_classification.py --config tests/test-configs/GraphTransformer_CYCLES_CYCLES_b25-bnorm-alt-trials-32-stoch-vary-softinit-initials200-topn.json --job_num 11 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;