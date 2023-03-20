# Bridging Graph Position Encodings for Transformers with Weighted Graph-Walking Automata

This repository contains the code for the paper ["Bridging Graph Position Encodings for Transformers with Weighted Graph-Walking Automata" (Soga and Chiang, 2023)](https://arxiv.org/abs/2212.06898). 

## Structure

There are 2 main directories in this repository. `graphs` contains code for running the graph dataset experiments, and `mt` contains code for running the machine translation experiments.

## Setup

To set up the environment for `graphs`, we recommend creating a new virtual environment with Python 3.10. Run

```bash
python -m venv ~/.virtualenvs/gape-graphs
```

To install dependencies for `graphs`, run the following:

1. `cd graphs`
2. `source ~/.virtualenvs/gape-graphs/bin/activate`
3. `pip3 install -r requirements.txt`

Dependencies for `mt` require an environment with Python 3.7 instead, which is easy to install separate from your main installation with Conda. Run the following after deactivating `gape-graphs`:

1. `cd mt`
2. `conda env create -f environment.yml`
3. `conda activate gape-mt`

Run the shell scripts in `graphs/data/` to download the graph datasets. The English-Vietnamese sentence pairs are already in `mt/nmt/data`.

## Running the Experiments

### Graphs

### MT

To verify the BLEU scores in the paper, run `python bleu.py OUT-[model]-[expected BLEU] nmt/data/en2vi/test.vi.tok` substituting the model name and the expected BLEU score. To train a model, run `train.sh` replacing the `PE` variable in `TASK=${SL}2${TL}_${PE}` with the desired PE scheme. See `nmt/configurations.py` for the list of PE schemes and their hyperparameters.

## References

Below is a table of all references and sources of code that is not ours.

| Reference | Code | Repository |
| --- | --- | --- |
| [Dwivedi et al. (2020)](https://arxiv.org/abs/2003.00982) | Graph transformer implementation, data preparation, and training & evaluation code | [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns) |
| [Dwivedi et al. (2022)](https://arxiv.org/abs/2110.07875) | Random walk PE implementation | [gnn-lspe](https://github.com/vijaydwivedi75/gnn-lspe) |
| [Kreuzer et al. (2021)](https://arxiv.org/abs/2106.03893) | Spectral attention node PE implementation | [SAN](https://github.com/DevinKreuzer/SAN)  |
| Nguyen | Transformer implementation for MT | [witwicky](https://github.com/tnq177/witwicky) | 
| [Ying et al. (2021)](https://arxiv.org/abs/2106.05234) | Shortest-path distance & centrality PE and Floyd-Warshall implementation | [Graphormer](https://github.com/microsoft/Graphormer)  |
