seed0=41
seed1=95
seed2=12
seed3=35

# no PE
log_file=PLANAR_graph_classification_no_pe.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-nope-trials.json --job_num 1 --pos_enc_dim 1 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# LAPE
log_file=PLANAR_graph_classification_lape.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-lape-20--trials.json --job_num 2 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# RW
log_file=PLANAR_graph_classification_rw.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-prwpe-20-trials.json --job_num 3 --pos_enc_dim 20 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SA
log_file=PLANAR_graph_classification_sa.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/SAGraphTransformer_Planarity_Planarity_b32-sa-500k-trials.json --job_num 4 --pos_enc_dim 10 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# SPD+C
log_file=PLANAR_graph_classification_spdc.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/PseudoGraphormer_Planarity_Planarity_b32-pseudo-500k-trials.json --job_num 5 --pos_enc_dim 1 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE
log_file=PLANAR_graph_classification_gape.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-rand-32-scale002-trials.json --job_num 6 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*
log_file=PLANAR_graph_classification_gape_stoch.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-rand-32-stoch-trials.json --job_num 7 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**
log_file=PLANAR_graph_classification_gape_stoch_softinit.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-rand-32-stoch-softinit-trials.json --job_num 8 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^*_20
log_file=PLANAR_graph_classification_gape_stoch_20.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-rand-32-stoch-initials20-trials.json --job_num 9 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_20
log_file=PLANAR_graph_classification_gape_stoch_softinit_20.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-rand-32-stoch-softinit-initials20-trials.json --job_num 10 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;

# GAPE^**_max
log_file=PLANAR_graph_classification_gape_stoch_softinit_max.log
python3 main_Planarity_graph_classification.py --config tests/test-configs/GraphTransformer_Planarity_Planarity_b32-rand-32-stoch-softinit-initials200-topn-trials.json --job_num 11 --pos_enc_dim 32 --log_file $log_file --seed_array $seed0 $seed1 $seed2 $seed3;