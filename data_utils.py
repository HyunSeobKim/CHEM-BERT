import re, os
import glob
import numpy as np
import pandas as pd
import random
import torch
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset

## Based on github.com/codertimo/BERT-pytorch

class Vocab(object):
	def __init__(self):
		self.pad_index = 0
		self.mask_index = 1
		self.unk_index = 2
		self.start_index = 3
		self.end_index = 4

		# check 'Na' later
		self.voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', \
							'(', 'n', '2', 'c', 'F', ')', '=', 'N', '3', 'S', '/', 's', '-', '+', 'o', 'P', \
							 'R', '\\', 'L', '#', 'X', '6', 'B', '7', '4', 'I', '5', 'i', 'p', '8', '9', '%', '0', '.', ':', 'A']

		self.dict = {s: i for i, s in enumerate(self.voca_list)}

	def __len__(self):
		return len(self.voca_list)

class SmilesDataset(Dataset):
	def __init__(self, smiles_path, vocab, seq_len, mat_position):
		self.vocab = vocab
		self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
		self.smiles_dataset = []
		self.adj_dataset = []
		self.seq_len = seq_len
		self.mat_pos = mat_position

		folder_list = os.listdir(smiles_path)

		for folder in folder_list:
			smiles_data = glob.glob(smiles_path + "/" + folder + "/*.smi")
			#print(smiles_data)
			for small_data in smiles_data:
				text = pd.read_csv(small_data, sep=" ")
				smiles_list = np.asarray(text['smiles'])
				for i in smiles_list:
					#adj_mat = GetAdjacencyMatrix(Chem.MolFromSmiles(i))
					#self.adj_dataset.append(self.zero_padding(adj_mat, (seq_len, seq_len)))
					self.adj_dataset.append(i)

					self.smiles_dataset.append(self.replace_halogen(i))

	def __len__(self):
		return len(self.smiles_dataset)

	def __getitem__(self, idx):
		item = self.smiles_dataset[idx]
		input_random, input_label, input_adj_mask = self.random_masking(item)

		input_data = [self.vocab.start_index] + input_random + [self.vocab.end_index]
		input_label = [self.vocab.pad_index] + input_label + [self.vocab.pad_index]
		input_adj_mask = [0] + input_adj_mask + [0]
		# give info to start token
		if self.mat_pos == 'start':
			input_adj_mask = [1] + [0 for _ in range(len(input_adj_mask)-1)]

		smiles_bert_input = input_data[:self.seq_len]
		smiles_bert_label = input_label[:self.seq_len]
		smiles_bert_adj_mask = input_adj_mask[:self.seq_len]

		padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
		smiles_bert_input.extend(padding)
		smiles_bert_label.extend(padding)
		smiles_bert_adj_mask.extend(padding)
		mol = Chem.MolFromSmiles(self.adj_dataset[idx])
		smiles_bert_value = QED.qed(mol)

		adj_mat = GetAdjacencyMatrix(mol)
		smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))

		output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": smiles_bert_label,  \
					"smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat, "smiles_bert_value": smiles_bert_value}

		return {key:torch.tensor(value) for key, value in output.items()}

	def random_masking(self,smiles):
		tokens = [i for i in smiles]
		output_label = []
		adj_masking = []

		for i, token in enumerate(tokens):
			if token in self.atom_vocab:
				adj_masking.append(1)
			else:
				adj_masking.append(0)

			prob = random.random()
			if prob < 0.15:
				prob /= 0.15

				if prob < 0.8:
					tokens[i] = self.vocab.mask_index

				# replace the token except special token
				elif prob < 0.9:
					tokens[i] = random.randrange(5,len(self.vocab))

				else:
					tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

				output_label.append(self.vocab.dict.get(token, self.vocab.unk_index))

			else:
				tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)
				output_label.append(self.vocab.pad_index) #modify to num

		return tokens, output_label, adj_masking

	def replace_halogen(self,string):
	    """Regex to replace Br,Cl,Sn,Na with single letters"""
	    br = re.compile('Br')
	    cl = re.compile('Cl')
	    sn = re.compile('Sn')
	    na = re.compile('Na')
	    string = br.sub('R', string)
	    string = cl.sub('L', string)
	    string = sn.sub('X', string)
	    string = na.sub('A', string)
	    return string

	def zero_padding(self, array, shape):
		padded = np.zeros(shape, dtype=np.float32)
		padded[:array.shape[0], :array.shape[1]] = array
		return padded

	def construct_vocab(self,smiles, vocab):
		smiles = replace_halogen(smiles)
		for i in smiles:
			if i not in vocab:
				vocab.append(i)
		return vocab

	def smiles_tokenizer(self,smiles):
		smiles = replace_halogen(smiles)
		char_list = [i for i in smiles]
		return char_list


def build_vocab(path):
	vocab = []
	folder_list = os.listdir(path)
	for folder in folder_list:
		smiles_data = glob.glob(path + "/"+folder+"/*.smi")
		for i in smiles_data:
			text = pd.read_csv(i, sep=" ")
			smiles_list = np.asarray(text['smiles'])
			for j in smiles_list:
				vocab = construct_vocab(j, vocab)
	print(vocab)

