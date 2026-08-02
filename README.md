# QAER: Query-Adaptive Evidence Routing for Temporal Knowledge Graph Extrapolation

This repository contains the official implementation of **QAER**, an evidence-aware method for temporal knowledge graph (TKG) extrapolation. Given a query `(subject, relation, ?, timestamp)`, QAER ranks every entity as a possible tail while exposing the historical evidence used by its scoring process.

## Repository Structure

```text
QAER_code_package/
|-- README.md                  # Documentation and commands
|-- requirements.txt           # Python dependencies
|-- config_presets/
|   `-- qaer.json              # Unified configuration for every dataset
|-- data/
|   |-- ICEWS14/
|   |-- ICEWS18/
|   |-- ICEWS05-15/
|   |-- GDELT/
|   |-- YAGO/
|   `-- WIKI/
|-- main.py                    # One-click train, validate, calibrate, and test pipeline
|-- configs.py                 # Configuration dataclasses and JSON loading
|-- data_loader.py             # Dataset loading and six-source evidence construction
|-- history_preprocess.py      # Automatic history preprocessing from raw quadruples
|-- semantic_encoder.py        # Pretrained semantic feature encoder
|-- temporal_graph.py          # Local temporal graph and path utilities
|-- model.py                   # QAER representation, routing, and scoring model
|-- train.py                   # Training, validation, EMA, and early stopping
|-- score_calibration.py       # Validation-only score calibration
|-- eval.py                    # Filtered full-entity evaluation and explanation export
`-- utils.py                   # Metrics, filtering, seeding, and file utilities
```

## Method Overview

QAER combines a structural-semantic neural scorer with explicit temporal evidence. Entity and relation representations fuse trainable structural embeddings with semantic features from `Qwen/Qwen3-Embedding-0.6B`. Subject history attention, time encoding, and a one-layer local R-GCN form the query representation.

For each query, QAER constructs six historical evidence sources:

- `copy`: previous occurrences of the same subject-relation-tail pattern;
- `local`: short-term subject-relation support;
- `rel`: relation-time tail frequency;
- `sub`: recent subject-candidate interactions;
- `global`: recent candidate-tail frequency;
- `path`: temporally valid reachability from the subject to the candidate.

A query-adaptive evidence router assigns weights to these sources. Their routed support is combined with the neural score while preserving full-entity ranking. The evaluation output records source-level score contributions and, when graph-derived path support is available, the corresponding timestamped historical paths.

## Environment

Python 3.9 and a CUDA-enabled PyTorch installation are recommended. Create an environment and install the dependencies with:

```powershell
conda create -n qaer python=3.9 -y
conda activate qaer
pip install -r requirements.txt
```

The default semantic encoder is downloaded from Hugging Face on first use. Set `semantic.local_files_only` to `true` in `config_presets/qaer.json` only when the model is already available in the local cache.

## Datasets

Every dataset directory contains the same six original files:

```text
data/<DATASET>/
|-- train.txt          # Training quadruples: subject relation object timestamp
|-- valid.txt          # Validation quadruples
|-- test.txt           # Test quadruples
|-- stat.txt           # Number of entities, relations, and temporal metadata
|-- entity2id.txt      # Entity-to-ID mapping
`-- relation2id.txt    # Relation-to-ID mapping
```

On the first run, `history_preprocess.py` automatically generates the required history files from the raw quadruples. Generated history files and semantic caches are ignored by Git.

| Dataset | Entities | Relations | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| ICEWS14 | 7,128 | 230 | 74,845 | 8,514 | 7,371 |
| ICEWS18 | 23,033 | 256 | 373,018 | 45,995 | 49,545 |
| ICEWS05-15 | 10,488 | 251 | 368,868 | 46,302 | 46,159 |
| GDELT | 7,691 | 240 | 1,734,399 | 238,765 | 305,241 |
| YAGO | 10,623 | 10 | 161,540 | 19,523 | 20,026 |
| WIKI | 12,554 | 24 | 539,286 | 67,538 | 63,110 |

GDELT is substantially larger than the other datasets and can be slow on a single consumer workstation.

## Evaluation

QAER uses filtered full-entity ranking and reports:

- **MRR**: mean reciprocal rank;
- **Hits@1**: proportion of targets ranked first;
- **Hits@3**: proportion of targets ranked in the top three;
- **Hits@10**: proportion of targets ranked in the top ten.

Other valid tails for the same `(subject, relation, timestamp)` query are removed before the filtered rank is computed.

## Unified Configuration

All six datasets use the same code and the same `config_presets/qaer.json`. Only the dataset name and output directory change between commands.

Key settings include:

| Field | Value |
|---|---:|
| Hidden dimension | 256 |
| Local R-GCN layers | 1 |
| History window | 96 |
| Path candidate cap | 64 |
| Routing temperature | 0.7 |
| Long-term evidence coefficient | 0.35 |
| Optimizer | AdamW |
| Learning rate | 0.0007 |
| Batch size | 512 |
| Maximum epochs | 20 |
| Validation frequency | every epoch |
| Early-stopping patience | 5 validations |
| Minimum MRR improvement | 0.0001 |

The best checkpoint is selected using validation MRR. Training stops when validation MRR fails to improve by more than `0.0001` for five consecutive validations, or after 20 epochs. Test labels are never used for checkpoint selection or calibration.

## One-Click Commands

Run each command from the repository root. The commands train QAER, validate after every epoch, fit validation-only calibration, and evaluate the selected checkpoint on the test set.

### ICEWS14

```powershell
python main.py --config config_presets/qaer.json --dataset-name ICEWS14 --output-dir outputs/ICEWS14/QAER --device cuda
```

### ICEWS18

```powershell
python main.py --config config_presets/qaer.json --dataset-name ICEWS18 --output-dir outputs/ICEWS18/QAER --device cuda
```

### ICEWS05-15

```powershell
python main.py --config config_presets/qaer.json --dataset-name ICEWS05-15 --output-dir outputs/ICEWS05-15/QAER --device cuda
```

### GDELT

```powershell
python main.py --config config_presets/qaer.json --dataset-name GDELT --output-dir outputs/GDELT/QAER --device cuda
```

### YAGO

```powershell
python main.py --config config_presets/qaer.json --dataset-name YAGO --output-dir outputs/YAGO/QAER --device cuda
```

### WIKI

```powershell
python main.py --config config_presets/qaer.json --dataset-name WIKI --output-dir outputs/WIKI/QAER --device cuda
```

For CPU execution, replace `--device cuda` with `--device cpu`. CPU training is considerably slower.

## Evaluation Only

After training, an existing checkpoint can be evaluated without retraining:

```powershell
python main.py --config config_presets/qaer.json --dataset-name ICEWS14 --output-dir outputs/ICEWS14/QAER --skip-train --checkpoint outputs/ICEWS14/QAER/best_model.pt --eval-split test --device cuda
```

## Generated Outputs

Each run creates the following files under the requested output directory:

```text
best_model.pt
resolved_config.json
training_log.json
score_calibration.json
test_metrics.json
test_predictions_with_paths.json
test_structured_evidence_chains.csv
```

`test_predictions_with_paths.json` contains ranked predictions, router weights, source-level score breakdowns, and available temporal-path evidence. Each saved prediction has a `temporal_paths` list whose entries contain node IDs, relation IDs, timestamps, the path score, normalized path-prior support, and a human-readable path. The same paths are grouped by candidate in `candidate_explanations`, while `path_recovery_audit` reports their output coverage. The CSV file stores the top prediction's paths in the `temporal_paths` column and renders them in readable form in `structured_evidence_chain`.

Explicit path reranking remains disabled during evaluation (`eval_path_topk=0`). After validation calibration and full-entity ranking determine the final Top-K predictions, QAER performs a score-neutral recovery pass over the same local historical graph used by path evidence. Recovery retains only edges before the query time, enforces nondecreasing timestamps along a path, and uses the configured path length, branch, and state limits. It changes neither candidate scores nor ranking metrics and is used only to serialize explanations.

## Publishing the Data Directory

The largest original file is `data/GDELT/train.txt` (about 39.5 MB). It is below GitHub's 100 MB per-file limit but exceeds the browser uploader's 25 MB limit. Publish the repository with Git command-line `add`, `commit`, and `push`, or use Git LFS; do not rely on the browser file uploader for the complete data directory.
