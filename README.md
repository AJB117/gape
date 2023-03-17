# Bridging Graph Position Encodings for Transformers with Weighted Graph-Walking Automata

This repository contains the code for the paper ["Bridging Graph Position Encodings for Transformers with Weighted Graph-Walking Automata" (Soga and Chiang, 2023)](https://arxiv.org/abs/2212.06898). We draw heavily on code from ["Benchmarking Graph Neural Networks" (Dwivedi et al., 2020)](https://arxiv.org/abs/2003.00982).

## Structure

There are 2 components to this repository. `graphs` contains code for running the graph dataset experiments, and `mt` contains code for running the machine translation experiments.

## Setup

We recommend creating 2 virtual environments for `graphs` and `mt` separately since `mt` relies on an older PyTorch version. To do so, run

```bash
python -m venv ~/.virtualenvs/gape-graphs; python -m venv ~/.virtualenvs/gape-mt
```

Run `pip3 install -r requirements.txt` in `graphs` and `mt` to install all dependencies.

## Running Code
