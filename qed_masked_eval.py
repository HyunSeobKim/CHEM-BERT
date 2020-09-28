import numpy as np 
import torch
import glob, os
import pandas as pd
import argparse
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.optim import Adam
import tqdm, re
from rdkit.Chem import QED
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from model import Smiles_BERT
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from data_utils import Vocab
from model import Smiles_BERT, Masked_prediction, BERT_base, BERT_double_tasks

class ADMETDataset(Dataset):
	def __init__(self, datapath, name, vocab, seq_len, trainType, mat_position):
		self.vocab = vocab
		self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
		self.smiles_dataset = []
		self.adj_dataset = []
		self.mat_pos = mat_position
		
		self.seq_len = seq_len
		path = datapath + "/" + name
		#print(path)
		smiles_data = glob.glob(path)
		text = pd.read_excel(smiles_data[0])

		text = text.loc[text['set'] == trainType]
		smiles_list = np.asarray(text['SMILES'])
		label_list = np.asarray(text['y'], dtype='float32')
		self.label = label_list.reshape(-1,1)
		for i in smiles_list:
			temp_i = Chem.MolToSmiles(Chem.MolFromSmiles(i))
			self.adj_dataset.append(temp_i)
			self.smiles_dataset.append(self.replace_halogen(temp_i))

	def __len__(self):
		return len(self.smiles_dataset)

	def __getitem__(self, idx):
		item = self.smiles_dataset[idx]
		#item = Chem.MolToSmiles(Chem.MolFromSmiles(i))
		input_random, input_label, input_adj_mask = self.random_masking(item)

		input_data = [self.vocab.start_index] + input_random + [self.vocab.end_index]
		input_label = [self.vocab.pad_index] + input_label + [self.vocab.pad_index]
		input_adj_mask = [0] + input_adj_mask + [0]
		if self.mat_pos == 'start':
			input_adj_mask = [1]

		smiles_bert_input = input_data[:self.seq_len]
		smiles_bert_label = input_label[:self.seq_len]
		smiles_bert_adj_mask = input_adj_mask[:self.seq_len]

		padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
		smiles_bert_input.extend(padding)
		smiles_bert_label.extend(padding)
		smiles_bert_adj_mask.extend(padding)
		mol = Chem.MolFromSmiles(self.adj_dataset[idx])
		adj_mat = GetAdjacencyMatrix(mol)
		smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))

		output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": smiles_bert_label,  \
					"smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat, "smiles_bert_value": QED.qed(mol)}

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

				if prob < 0.80:
					tokens[i] = self.vocab.mask_index

				# replace the token except special token
				elif prob < 0.90:
					tokens[i] = random.randrange(5,len(self.vocab))

				else:
					tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)
				output_label.append(self.vocab.dict.get(token, self.vocab.unk_index))

			else:
				tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)
				output_label.append(self.vocab.pad_index) #modify to num

		return tokens, output_label, adj_masking

	def CharToNum(self, smiles):
		tokens = [i for i in smiles]
		adj_masking = []

		for i, token in enumerate(tokens):
			if token in self.atom_vocab:
				adj_masking.append(1)
			else:
				adj_masking.append(0)

			tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

		return tokens, adj_masking


	def replace_halogen(self,string):
	    """Regex to replace Br and Cl with single letters"""
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
		if array.shape[0] > shape[0]:
			array = array[:shape[0],:shape[1]]
		padded = np.zeros(shape, dtype=np.float32)
		padded[:array.shape[0], :array.shape[1]] = array
		return padded

class SmilesDataset(Dataset):
	def __init__(self, smiles_path, name, vocab, seq_len, mat_position):
		self.vocab = vocab
		self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
		self.smiles_dataset = []
		self.adj_dataset = []
		self.seq_len = seq_len
		self.mat_pos = mat_position

		
		smiles_data = glob.glob(smiles_path + "/" + name)
		#print(smiles_data)
		
		text = pd.read_csv(smiles_data[0], sep=" ")
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
		if self.mat_pos == 'start':
			input_adj_mask = [1]

		smiles_bert_input = input_data[:self.seq_len]
		smiles_bert_label = input_label[:self.seq_len]
		smiles_bert_adj_mask = input_adj_mask[:self.seq_len]

		padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
		smiles_bert_input.extend(padding)
		smiles_bert_label.extend(padding)
		smiles_bert_adj_mask.extend(padding)

		adj_mat = GetAdjacencyMatrix(Chem.MolFromSmiles(self.adj_dataset[idx]))
		smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))

		output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": smiles_bert_label,  \
					"smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat}

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
	    """Regex to replace Br and Cl with single letters"""
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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help="dataset path", type=str, default = None)
	parser.add_argument('--name', help="name of dataset", type=str, default=None)
	parser.add_argument('--data_indice', help="indices of dataset", type=str, default=None)
	parser.add_argument('--adjacency', help="use adjacency matrix", type=bool, default=False)
	parser.add_argument('--batch', help="batch size", type=int, default=128)
	parser.add_argument('--embed_size', help="embedding vector size", type=int, default=1024)
	parser.add_argument('--seq', help="sequence length", type=int, default=256)
	parser.add_argument('--layers', help="number of layers", type=int, default=6)
	parser.add_argument('--nhead', help="number of head", type=int, default=4)
	parser.add_argument('--saved_model', help="dir of fine-tuned model", type=str)
	parser.add_argument('--matrix_position', help="position of adjacency matrix", type=str, default='atom')
	parser.add_argument('--num_workers', help="number of workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument('--type', type=str)
	#parser.add_argument('--type', help="type of dataset", type=str)
	arg = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#device = torch.device("cpu")
	print("device:", device)
	Smiles_vocab = Vocab()
	if arg.type == 'zinc':
		testdataset = SmilesDataset(arg.path, Smiles_vocab, seq_len=arg.seq, mat_position=arg.matrix_position)
	else:
		testdataset = ADMETDataset(arg.path, arg.name, Smiles_vocab, seq_len=arg.seq, trainType='Training', mat_position=arg.matrix_position)
	test_dataloader = DataLoader(testdataset, batch_size=arg.batch, num_workers=arg.num_workers)

	model = Smiles_BERT(len(Smiles_vocab), max_len=arg.seq, nhead=arg.nhead, model_dim=arg.embed_size, nlayers=arg.layers, adj=arg.adjacency)
	value_layer = nn.Linear(arg.embed_size, 1)
	mask_layer = Masked_prediction(arg.embed_size, len(Smiles_vocab))
	model = BERT_double_tasks(model, value_layer, mask_layer)

	model.load_state_dict(torch.load(arg.saved_model))
	model.to(device)
	#if torch.cuda.device_count() > 1:
	#	model = nn.DataParallel(model)


	correct = 0
	total = 0
	predicted_list = np.array([])
	target_list = np.array([])
	total_loss = 0

	criterion = nn.L1Loss()

	model.eval()
	test_iter = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
	position_num = torch.arange(arg.seq).repeat(arg.batch,1).to(device)

	with torch.no_grad():
		for i, data in test_iter:
			data = {key:value.to(device) for key, value in data.items()}
			if data["smiles_bert_input"].size(0) != arg.batch:
				position_num = torch.arange(arg.seq).repeat(data["smiles_bert_input"].size(0),1).to(device)
			if arg.adjacency is True:
				qed_output, output = model(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
			else:
				qed_output, output = model(data["smiles_bert_input"], position_num)
			#output = output[:,0]
			loss = criterion(qed_output, data["smiles_bert_value"].view(-1,1))
			total_loss += loss.item()
			predicted = output.argmax(dim=-1)
			#print(predicted, data["smiles_bert_label"].shape)
			for k in range(predicted.size(0)):
				for j in range(predicted.size(1)):
					if data["smiles_bert_label"][k][j].item() != 0:
						correct += predicted[k][j].eq(data["smiles_bert_label"][k][j].item()).sum().item()
						total += 1

			#predicted_list = np.append(predicted_list, predicted.cpu().detach().numpy())
			#target_list = np.append(target_list, data["smiles_bert_label"].cpu().detach().numpy())
			#_, predicted = torch.max(output.data, 1)

			#total += data["smiles_bert_label"].size(0)
			#correct += (torch.round(predicted) == data["smiles_bert_label"]).sum().item()

	#predicted_list = np.reshape(predicted_list, (-1))
	#target_list = np.reshape(target_list, (-1))
	#print(predicted_list, target_list)
	print("Accuracy on testset: ", 100 * correct / total, "MAE on QED:", total_loss / len(test_iter))



if __name__ == "__main__":
	main()