import numpy as np 
import torch
import os, re, time
import glob
import pandas as pd
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, SequentialSampler

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, auc
from model import Smiles_BERT, BERT_base

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

class FinetuningDataset(Dataset):
	def __init__(self, datapath, vocab, seq_len):
		self.vocab = vocab
		self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
		self.smiles_dataset = []
		self.adj_dataset = []
		
		self.seq_len = seq_len

		smiles_data = glob.glob(datapath)
		text = pd.read_csv(smiles_data[0])

		csv_columns = text.columns
		if len(csv_columns) == 2:
			try:
				if Chem.MolFromSmiles(text[csv_columns[0]][0]) != None and type(text[csv_columns[0]][0]) == str:
					smiles_list = np.asarray(text[csv_columns[0]])
					label_list = np.asarray(text[csv_columns[1]])
				else:
					smiles_list = np.asarray(text[csv_columns[1]])
					label_list = np.asarray(text[csv_columns[0]])
			except:
				smiles_list = np.asarray(text[csv_columns[1]])
				label_list = np.asarray(text[csv_columns[0]])
		else:
			raise NameError("The number of columns should be two (smiles and y_label).")
			print("The number of columns should be two. (smiles and y)")
			exit(1)
		'''
		try:
			smiles_list = np.asarray(text['smiles'])
			label_list = np.asarray(text['y'])
		except:
			print("Header should include smiles and y")
			exit(1)
		'''

		self.label = label_list.reshape(-1,1)
		for i in smiles_list:
			self.adj_dataset.append(i)
			self.smiles_dataset.append(self.replace_halogen(i))

	def __len__(self):
		return len(self.smiles_dataset)

	def __getitem__(self, idx):
		item = self.smiles_dataset[idx]
		label = self.label[idx]

		input_token, input_adj_masking = self.CharToNum(item)

		input_data = [self.vocab.start_index] + input_token + [self.vocab.end_index]
		input_adj_masking = [0] + input_adj_masking + [0]

		smiles_bert_input = input_data[:self.seq_len]
		smiles_bert_adj_mask = input_adj_masking[:self.seq_len]

		padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
		smiles_bert_input.extend(padding)
		smiles_bert_adj_mask.extend(padding)

		mol = Chem.MolFromSmiles(self.adj_dataset[idx])
		if mol != None:
			adj_mat = GetAdjacencyMatrix(mol)
			smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))
		else:
			smiles_bert_adjmat = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)

		output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": label,  \
					"smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat}

		return {key:torch.tensor(value) for key, value in output.items()}

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

def main():
	with open('input/input.json', 'r') as f:
		input_file = json.load(f)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available() == False:
		torch.set_num_threads(24)
	params = {'batch_size':64, 'dropout':0, 'learning_rate':0.00001, 'epoch':15, 'optimizer':'Adam', 'model':'Transformer'}

	Smiles_vocab = Vocab()
	# read data
	dataset = FinetuningDataset('input/dataset.csv', Smiles_vocab, seq_len=256)

	indices = list(range(len(dataset)))
	np.random.shuffle(indices)
	split1, split2 = int(np.floor(input_file['split_ratio'] * len(dataset))), int(np.floor(0.5 * (1 + input_file['split_ratio']) * len(dataset)))
	train_idx, valid_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)
	test_sampler = SubsetRandomSampler(test_idx)

	# dataloader(train, valid, test)
	train_dataloader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
	valid_dataloader = DataLoader(dataset, batch_size=params['batch_size'], sampler=valid_sampler, num_workers=4)
	test_dataloader = DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler, num_workers=4)

	#Load model
	model = Smiles_BERT(len(Smiles_vocab), max_len=256, nhead=16, feature_dim=1024, feedforward_dim=1024, nlayers=8, adj=True, dropout_rate=params['dropout'])
	model.load_state_dict(torch.load('../model/pretrained_model.pt', map_location=device))
	output_layer = nn.Linear(1024, 1)

	model = BERT_base(model, output_layer)
	model.to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	optim = Adam(model.parameters(), lr=params['learning_rate'], weight_decay=0)
	if input_file['task'] == 'classification':
		criterion = nn.BCEWithLogitsLoss()
	else:
		criterion = nn.MSELoss()

	test_crit = nn.MSELoss(reduction='none')

	train_loss_list = []
	valid_loss_list = []
	valid_score_list = []
	test_score_list = []
	test_result_list = []

	min_valid_loss = 10000
	start_time = time.time()

	#Start training
	for epoch in range(params['epoch']):
		avg_loss = 0
		valid_avg_loss = 0
		valid_rmse = []
		test_rmse = []
		test_pred = []
		test_true = []
		predicted_list = np.array([])
		target_list = np.array([])

		model.train()
		for i, data in enumerate(train_dataloader):
			data = {key:value.to(device) for key, value in data.items()}
			position_num = torch.arange(256).repeat(data["smiles_bert_input"].size(0),1).to(device)
			output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
			output = output[:,0].double()

			loss = criterion(output, data["smiles_bert_label"])
			optim.zero_grad()
			loss.backward()
			optim.step()

			avg_loss += loss.item()
		train_loss_list.append(avg_loss / len(train_dataloader))

		model.eval()
		with torch.no_grad():
			#validation set
			for i, data in enumerate(valid_dataloader):
				data = {key:value.to(device) for key, value in data.items()}
				position_num = torch.arange(256).repeat(data["smiles_bert_input"].size(0),1).to(device)
				output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
				output = output[:,0]
				valid_loss = criterion(output, data["smiles_bert_label"])
				valid_avg_loss += valid_loss.item()

				if input_file['task'] == 'classification':
					predicted = torch.sigmoid(output)
					predicted_list = np.append(predicted_list, predicted.cpu().detach().numpy())
					target_list = np.append(target_list, data["smiles_bert_label"].cpu().detach().numpy())
				else:
					test_loss = torch.sqrt(test_crit(output, data["smiles_bert_label"]))
					valid_rmse.append(test_loss)

			valid_avg_loss = valid_avg_loss / len(valid_dataloader)
			valid_loss_list.append(valid_avg_loss)

			if input_file['task'] == 'classification':
				predicted_list = np.reshape(predicted_list, (-1))
				target_list = np.reshape(target_list, (-1))
				valid_score_list.append(roc_auc_score(target_list, predicted_list))
			else:
				valid_rmse = torch.cat(valid_rmse, dim=0).cpu().numpy()
				valid_rmse = sum(valid_rmse) / len(valid_rmse)
				valid_score_list.append(valid_rmse[0])

			#save the model
			if valid_avg_loss < min_valid_loss:
				save_path = 'output/derived/Finetuned_model.pt'
				torch.save(model.module.state_dict(), save_path)
				model.to(device)
				min_valid_loss = valid_avg_loss

			predicted_list = np.array([])
			target_list = np.array([])

			#Test set
			for i, data in enumerate(test_dataloader):
				data = {key:value.to(device) for key, value in data.items()}
				position_num = torch.arange(256).repeat(data["smiles_bert_input"].size(0),1).to(device)
				output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
				output = output[:,0]

				if input_file['task'] == 'classification':
					predicted = torch.sigmoid(output)
					predicted_list = np.append(predicted_list, predicted.cpu().detach().numpy())
					target_list = np.append(target_list, data["smiles_bert_label"].cpu().detach().numpy())
				else:
					test_loss = torch.sqrt(test_crit(output, data["smiles_bert_label"]))
					test_rmse.append(test_loss)
					test_pred.append(output)
					test_true.append(data["smiles_bert_label"])

			if input_file['task'] == 'classification':
				predicted_list = np.reshape(predicted_list, (-1))
				target_list = np.reshape(target_list, (-1))
				test_result_list.append((target_list, predicted_list))
				test_score_list.append(roc_auc_score(target_list, predicted_list))
			else:
				test_rmse = torch.cat(test_rmse, dim=0).cpu().numpy()
				test_pred = torch.cat(test_pred, dim=0).cpu().numpy()
				test_true = torch.cat(test_true, dim=0).cpu().numpy()
				test_result_list.append((test_true, test_pred))
				test_rmse = sum(test_rmse) / len(test_rmse)
				test_score_list.append(test_rmse[0])

		end_time = time.time()
		if end_time - start_time > input_file['time'] * 60:
			break

	#plot the graph & return best score
	plt.figure(1)
	x_len = np.arange(len(train_loss_list))
	plt.plot(x_len, train_loss_list, marker='.', c='blue', label="Train loss")
	plt.plot(x_len, valid_loss_list, marker='.', c='red', label="Validation loss")
	plt.legend(loc='upper right')
	plt.grid()
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.title('Loss graph')
	plt.savefig('output/derived/loss_graph.png')

	output_csv = pd.DataFrame()
	output_csv['Train_loss'] = train_loss_list
	output_csv['Valid_loss'] = valid_loss_list
	output_csv['Valid_score'] = valid_score_list
	output_csv.to_csv('output/derived/result.csv')

	best_step = np.argmin(valid_loss_list)
	best_score = test_score_list[best_step]

	# roc curve or r2


	output_json = params
	if input_file['task'] == 'classification':
		output_json['metric'] = 'AUC-ROC'
		fpr, tpr, _ = roc_curve(test_result_list[best_step][0], test_result_list[best_step][1])
		plt.figure(2)
		lw = 2
		plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % auc(fpr, tpr))
		plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc='lower right')
		plt.savefig('output/derived/test_score.png')

	else:
		output_json['metric'] = 'RMSE'
		plt.figure(3)
		t_true, t_pred = test_result_list[best_step][0], test_result_list[best_step][1]
		plt.scatter(t_true, t_pred, s=10)
		plt.plot([t_true.min(), t_true.max()], [t_true.min(), t_true.max()], color='red', label='R-squared Score = %0.2f' % r2_score(t_true, t_pred), linestyle='--')
		plt.xlabel('Actual')
		plt.ylabel('Predicted')
		plt.title('Scatter Plot')
		plt.legend(loc='lower right')
		plt.savefig('output/derived/test_score.png')
	output_json['best_score'] = round(best_score,5)

	with open('output/metadata/dm.json', 'w') as f:
		json.dump(output_json, f)

if __name__ == "__main__":
	if os.path.isdir('output') == False:
		os.mkdir('output')
	if os.path.isdir('output/derived') == False:
		os.mkdir('output/derived')
	if os.path.isdir('output/metadata') == False:
		os.mkdir('output/metadata')
	if os.path.isdir('output/log') == False:
		os.mkdir('output/log')
	main()