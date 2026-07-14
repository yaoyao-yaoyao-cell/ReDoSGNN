# ReDoS-MotifGNN (RMGNN)

This repository contains the reference implementation of **RMGNN**, the model
introduced in *Exploring Motif-based Heterogeneous Graph Learning for ReDoS
Detection* (ICML 2026).


## Repository layout

```text
.
├── main.py                         # Stable training entry point
├── preprocess.py                   # Stable preprocessing entry point
├── requirements.txt
├── RMGNN_ICML_Version.pdf
├── dataset/
│   ├── OutputAST/                  # Raw regex AST dumps
│   └── *_HRG/                      # TU-style HRG datasets
└── rmgnn/
    ├── training.py                 # Cross-validation and evaluation
    ├── preprocessing.py            # Motif extraction and meta-graph building
    ├── data/
    │   ├── ast_converter.py        # AST-to-HRG semantic conversion
    │   ├── regex_graph_loader.py   # HRG loading and graph data structures
    │   ├── legacy_format.py        # Optional legacy text conversion
    │   └── legacy_loader.py        # Legacy experiment data loader
    └── models/
        ├── rmgnn.py                # Main RMGNN classifier
        ├── heterogeneous_conv.py   # Relation-aware weighted propagation
        ├── graph_encoder.py        # Local HRG encoder
        ├── mlp.py                  # Shared MLP block
        └── baselines/              # DGL and PyG comparison models
```

## Installation

Python 3.10 or newer is recommended. Install PyTorch for the CUDA version used
by your system, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The optional models under `rmgnn/models/baselines/` additionally require
PyTorch Geometric, `torch-scatter`, and OGB. They are not needed for RMGNN.

## Data preparation

Preprocess an existing TU-style HRG dataset:

```bash
python preprocess.py --data Corpus_HRG
```

Convert raw OutputAST files and preprocess them in one command:

```bash
python preprocess.py --data Corpus --convert --overwrite
```

Use `--data all` for all four paper datasets. For a quick pipeline check, use
`--sample-size`, `--max-instances-per-graph`, and `--skip-teacher`.

## Training and evaluation

Run the paper-style stratified cross-validation protocol:

```bash
python main.py --data Corpus_HRG --num-epochs 100 --folds 10
```

Run a short smoke experiment:

```bash
python main.py --data Corpus_HRG --num-epochs 1 --fold-limit 1
```

Results are saved as timestamped JSON files under `result/`. Run
`python main.py --help` and `python preprocess.py --help` for all options.

## Dataset format

Each `dataset/<name>_HRG/` directory uses the TU graph format:

- `A.txt`: graph edges;
- `graph_indicator.txt`: graph membership of each node;
- `graph_labels.txt`: binary ReDoS labels;
- `node_labels.txt` and `node_attributes.txt`: node types and semantic features;
- `edge_labels.txt`: syntax and semantic relation types.

Labels use `0` for safe regexes and `1` for ReDoS-vulnerable regexes after
loading. See `rmgnn/data/ast_converter.py` for the AST input convention.

## Citation

Please cite the ICML 2026 paper if you use this code. Replace this section with
the final proceedings BibTeX entry when it becomes available.
