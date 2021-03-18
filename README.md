# A Merged Molecular Representation Learning

Pre-training BERT with molecular data for learning merged molecular representation.

Demo Video: https://youtu.be/efDLkbdXvRg

## Usage for pre-training

Make a directory for saving models.

After preparing data with SMILES format (e.g. ZINC) and modifying SmilesDataset in data_utils.py, run the pretraining.py or double_pretraining.py.

`python double_pretraining.py --path {dataset_path} --save_path {model_path} --adjacency True --batch 256 --epoch 20 --seq 256 --layers 6 --nhead 16 --seed 7`

## Fine-tuning with a pre-trained model

Download the pre-trained model using `git lfs pull`.

For using the fine-tuning service, prepare an input and output directory.

A dataset with CSV format and input.json should be located in the input directory.

### Example input file for the fine-tuning service

input.json - {"split-ratio":0.8, "task":classification, "time":30}

dataset.csv - 2 columns (smiles, label)
