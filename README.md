# MergedBERT

Pre-training BERT with molecular data

## Usage

After preparing data with SMILES format (e.g. ZINC), run the pretraining.py or double_pretraining.py

For using the fine-tuning service, prepare input, output folder in your path.

In input folder, dataset with csv format and input.json should be located.

### Example input file for the fine-tuning service

input.json - {"split-ratio":0.8, "task":classification, "time":30}

dataset.csv - 2 columns (smiles, label)
