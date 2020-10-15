# MergedBERT

Pre-training BERT with molecular data for merged molecular representation.

## Usage for pretraining

After preparing data with SMILES format (e.g. ZINC) and modifying SmilesDataset in data_utils.py, run the pretraining.py or double_pretraining.py.

`python double_pretraining.py --path {dataset_path} --adjacency True --batch 256 --epoch 20 --seq 256 --layers 6 --nhead 16 --seed 7`

## Fine-tuning with a pretrained model

For using the fine-tuning service, prepare input, output folder in your path.

In input folder, dataset with csv format and input.json should be located.

### Example input file for the fine-tuning service

input.json - {"split-ratio":0.8, "task":classification, "time":30}

dataset.csv - 2 columns (smiles, label)
