import numpy as np 
import torch
import glob
import pandas as pd
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm, re
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from model import Smiles_BERT
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, SequentialSampler
from data_utils import Vocab
from model import Smiles_BERT, Masked_prediction, Smiles_BERT_BC, BERT_base, BERT_add_feature, BERT_base_dropout
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

def generate_scaffold(smiles, include_chirality=False):
	scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
	return scaffold

def scaffold_split(smiles):
	all_scaffolds = {}
	count = 0
	for i in range(len(smiles)):
		if Chem.MolFromSmiles(smiles[i]) != None:
			scaffold = generate_scaffold(smiles[i], include_chirality=True)
			if scaffold not in all_scaffolds:
				all_scaffolds[scaffold] = [i]
			else:
				all_scaffolds[scaffold].append(i)
		else:
			count += 1
	all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
	all_scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(all_scaffolds.items(), 
						key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
	train_cutoff = 0.8 * (len(smiles)-count)
	valid_cutoff = 0.9 * (len(smiles)-count)
	train_idx, valid_idx, test_idx = [], [], []
	for scaffold_set in all_scaffold_sets:
		if len(train_idx) + len(scaffold_set) > train_cutoff:
			if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
				test_idx.extend(scaffold_set)
			else:
				valid_idx.extend(scaffold_set)
		else:
			train_idx.extend(scaffold_set)
	assert len(set(train_idx).intersection(set(valid_idx))) == 0
	assert len(set(test_idx).intersection(set(valid_idx))) == 0

	return train_idx, valid_idx, test_idx

def random_scaffold(smiles, seed):
	rng = np.random.RandomState(seed)
	all_scaffolds = {}
	count = 0
	for i in range(len(smiles)):
		if Chem.MolFromSmiles(smiles[i]) != None:
			scaffold = generate_scaffold(smiles[i], include_chirality=True)
			if scaffold not in all_scaffolds:
				all_scaffolds[scaffold] = [i]
			else:
				all_scaffolds[scaffold].append(i)
		else:
			count += 1

	all_scaffolds = rng.permutation(list(all_scaffolds.values()))
	all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
	all_scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(all_scaffolds.items(), 
						key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
	train_cutoff = 0.8 * (len(smiles)-count)
	valid_cutoff = 0.9 * (len(smiles)-count)
	train_idx, valid_idx, test_idx = [], [], []
	for scaffold_set in all_scaffold_sets:
		if len(train_idx) + len(scaffold_set) > train_cutoff:
			if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
				test_idx.extend(scaffold_set)
			else:
				valid_idx.extend(scaffold_set)
		else:
			train_idx.extend(scaffold_set)
	assert len(set(train_idx).intersection(set(valid_idx))) == 0
	assert len(set(test_idx).intersection(set(valid_idx))) == 0

	return train_idx, valid_idx, test_idx


def _init_seed_fix(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def add_descriptors(mol):
    feature_List = []
    c_patt = Chem.MolFromSmiles('C')
    o_patt = Chem.MolFromSmiles('O')
    n_patt = Chem.MolFromSmiles('N')
    cl_patt = Chem.MolFromSmiles('Cl')
    
    try :
        #mol = Chem.MolFromSmiles(smile)
        ### aLogp, MW, HBA, HBD, RotB, PSA
        qedp = QED.properties(mol)
        # aLogP
        feature_List.append(qedp.ALOGP)
        # MW : mol Weight
        # MW = m_des.CalcExactMolWt(mol)
        feature_List.append(qedp.MW)
        # HBA/HBD : HB acceptor and donor
        # HBA=m_des.CalcNumHBA(mol)
        # HBD=m_des.CalcNumHBD(mol)
        feature_List.append(qedp.HBA)
        feature_List.append(qedp.HBD)
        # RotB : Rotatable bonds
        # RotB = Lipinski.NumRotatableBonds(mol)
        feature_List.append(qedp.ROTB)
        # PoarSurfaceArea
        feature_List.append(qedp.PSA)
        
        Rings = Lipinski.RingCount(mol)
        feature_List.append(Rings)
        
        # AroRing
        AroRing = Lipinski.NumAromaticRings(mol)
        feature_List.append(AroRing)
        
        # Atoms
        NAtoms = mol.GetNumAtoms()
        feature_List.append(NAtoms)


        ###
        feature_List.append(len(mol.GetSubstructMatches(c_patt)))
        feature_List.append(len(mol.GetSubstructMatches(o_patt)))
        feature_List.append(len(mol.GetSubstructMatches(n_patt)))
        feature_List.append(len(mol.GetSubstructMatches(cl_patt)))
        ###
        feature_List.append(des.NumValenceElectrons(mol))
        feature_List.append(des.MaxAbsPartialCharge(mol))
        feature_List.append(des.MinAbsPartialCharge(mol))
        
        feature_List.append(des.MolLogP(mol))
        feature_List.append(des.MolMR(mol))
        feature_List.append(des.MolWt(mol))
        feature_List.append(des.NHOHCount(mol))
        feature_List.append(des.NOCount(mol))
        
        feature_List.append(des.NumAliphaticCarbocycles(mol))
        feature_List.append(des.NumAliphaticHeterocycles(mol))
        feature_List.append(des.NumAliphaticRings(mol))
        feature_List.append(des.NumAromaticCarbocycles(mol))
        feature_List.append(des.NumAromaticHeterocycles(mol))
        feature_List.append(des.NumAromaticRings(mol))
        
        feature_List.append(des.NumHAcceptors(mol))
        feature_List.append(des.NumHDonors(mol))
        feature_List.append(des.NumHeteroatoms(mol))
        
        feature_List.append(des.NumRadicalElectrons(mol))
        feature_List.append(des.NumRotatableBonds(mol))
        feature_List.append(des.NumSaturatedCarbocycles(mol))
        feature_List.append(des.NumSaturatedHeterocycles(mol))
        feature_List.append(des.NumSaturatedRings(mol))
        feature_List.append(mol.GetNumHeavyAtoms())
        feature_List.append(mol.GetNumBonds())
        
        ### col list for 256bit ECFP6 column name
        
        # ECFP_6
        
    except ValueError : pass 
    
    return feature_List

class FinetuningDataset(Dataset):
	def __init__(self, datapath, data_name, vocab, seq_len, trainType, mat_position):
		self.vocab = vocab
		self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
		self.smiles_dataset = []
		self.adj_dataset = []
		self.mat_pos = mat_position
		
		self.seq_len = seq_len

		smiles_data = glob.glob(datapath + "/" + data_name + ".csv")
		text = pd.read_csv(smiles_data[0], sep=',')
		if data_name == 'tox21':
			tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
	       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
			#text = text.loc[text['set'] == trainType]
			smiles_list = np.asarray(text['smiles'])
			label_list = text[tasks]
			label_list = label_list.replace(0,-1)
			label_list = label_list.fillna(0)
			#label_list = np.asarray(label_list)
		elif data_name == 'bace':
			smiles_list = np.asarray(text['smiles'])
			label_list = text['Class']
			label_list = label_list.replace(0, -1)
		elif data_name == 'bbbp':
			smiles_list = np.asarray(text['smiles'])
			label_list = text['p_np']
			label_list = label_list.replace(0, -1)
		elif data_name == 'clintox':
			smiles_list = np.asarray(text['smiles'])
			tasks = ['FDA_APPROVED', 'CT_TOX']
			label_list = text[tasks]
			label_list = label_list.replace(0, -1)
		elif data_name == 'sider':
			smiles_list = np.asarray(text['smiles'])
			tasks = ['Hepatobiliary disorders',
       				'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
				    'Investigations', 'Musculoskeletal and connective tissue disorders',
				    'Gastrointestinal disorders', 'Social circumstances',
				    'Immune system disorders', 'Reproductive system and breast disorders',
				    'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
				    'General disorders and administration site conditions',
				    'Endocrine disorders', 'Surgical and medical procedures',
				    'Vascular disorders', 'Blood and lymphatic system disorders',
				    'Skin and subcutaneous tissue disorders',
				    'Congenital, familial and genetic disorders',
				    'Infections and infestations',
				    'Respiratory, thoracic and mediastinal disorders',
				    'Psychiatric disorders', 'Renal and urinary disorders',
				    'Pregnancy, puerperium and perinatal conditions',
				    'Ear and labyrinth disorders', 'Cardiac disorders',
				    'Nervous system disorders',
				    'Injury, poisoning and procedural complications']
			label_list = text[tasks]
			label_list = label_list.replace(0, -1)
		elif data_name == 'toxcast':
			smiles_list = np.asarray(text['smiles'])
			tasks = list(text.columns[1:])
			label_list = text[tasks]
			label_list = label_list.replace(0, -1)
			label_list = label_list.fillna(0)
		elif data_name == 'muv':
			smiles_list = np.asarray(text['smiles'])
			tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
					'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
					'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
			label_list = text[tasks]
			label_list = label_list.replace(0, -1)
			label_list = label_list.fillna(0)
		elif data_name == 'hiv':
			smiles_list = np.asarray(text['smiles'])
			label_list = text['HIV_active']
			label_list = label_list.replace(0, -1)

		self.label = np.asarray(label_list)
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
		if self.mat_pos == 'start':
			input_adj_mask = [1] + [0 for _ in range(len(input_adj_mask)-1)]

		smiles_bert_input = input_data[:self.seq_len]
		smiles_bert_adj_mask = input_adj_masking[:self.seq_len]

		padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
		smiles_bert_input.extend(padding)
		smiles_bert_adj_mask.extend(padding)

		mol = Chem.MolFromSmiles(self.adj_dataset[idx])
		#features = add_descriptors(mol)
		#smiles_bert_ECFP = np.array(features, dtype=np.float32)
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
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help="dataset path", type=str, default = None)
	parser.add_argument('--dataset', help="name of dataset", type=str)
	#parser.add_argument('--data_indice', help="indices of dataset", type=str)
	parser.add_argument('--adjacency', help="use adjacency matrix", type=bool, default=False)
	parser.add_argument('--batch', help="batch size", type=int, default=128)
	parser.add_argument('--epoch', help="epoch", type=int, default=100)
	parser.add_argument('--seq', help="sequence length", type=int, default=256)
	parser.add_argument('--lr', help="learning rate", type=float, default=0.0001)
	parser.add_argument('--embed_size', help="embedding vector size", type=int, default=1024)
	parser.add_argument('--model_dim', help="dim of transformer", type=int, default=1024)
	parser.add_argument('--layers', help="number of layers", type=int, default=6)
	parser.add_argument('--nhead', help="number of head", type=int, default=4)
	parser.add_argument('--drop_rate', help="ratio of dropout", type=float, default=0)
	parser.add_argument('--matrix_position', help="position of adjacency matrix", type=str, default='atom')
	parser.add_argument('--warmup_step', help="warmup step for scheduled learning rate", type=int, default=10000)
	parser.add_argument('--num_workers', help="number of workers", type=int, default=0)
	parser.add_argument('--split', help="type of dataset", type=str, default='scaffold')
	parser.add_argument('--saved_model', help="dir of pre-trained model", type=str)
	parser.add_argument("--seed", type=int, default=7)
	arg = parser.parse_args()

	_init_seed_fix(arg.seed)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("device:", device)

	if arg.dataset == "tox21":
		num_tasks = 12
	elif arg.dataset == "bace":
		num_tasks = 1
	elif arg.dataset == 'bbbp':
		num_tasks = 1
	elif arg.dataset == 'clintox':
		num_tasks = 2
	elif arg.dataset == 'sider':
		num_tasks = 27
	elif arg.dataset == 'toxcast':
		num_tasks = 617
	elif arg.dataset == 'muv':
		num_tasks = 17
	elif arg.dataset == 'hiv':
		num_tasks = 1

	Smiles_vocab = Vocab()
	# read data
	dataset = FinetuningDataset(arg.path, arg.dataset, Smiles_vocab, seq_len=arg.seq, trainType='Training', mat_position=arg.matrix_position)
	print("Dataset loaded")
	if arg.split == 'scaffold':
		smiles_csv = pd.read_csv(arg.path+"/"+arg.dataset+".csv", sep=',')
		smiles_list = smiles_csv['smiles'].tolist()
		
		train_idx, valid_idx, test_idx = scaffold_split(smiles_list)
	elif arg.split == 'random_scaffold':
		smiles_list = smiles_csv['smiles'].tolist()
		
		train_idx, valid_idx, test_idx = random_scaffold(smiles_list, arg.seed)
	else:
		indices = list(range(len(dataset)))
		split1, split2 = int(np.floor(0.1 * len(dataset))), int(np.floor(0.2 * len(dataset)))
		#np.random.seed(arg.seed)
		np.random.shuffle(indices)
		train_idx, valid_idx, test_idx = indices[split2:], indices[split1:split2], indices[:split1]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)
	test_sampler = SubsetRandomSampler(test_idx)

	# preprocessing - dataloader(train, valid, test)
	train_dataloader = DataLoader(dataset, batch_size=arg.batch, sampler=train_sampler, num_workers=arg.num_workers, pin_memory=True)
	valid_dataloader = DataLoader(dataset, batch_size=arg.batch, sampler=valid_sampler, num_workers=arg.num_workers)
	test_dataloader = DataLoader(dataset, batch_size=arg.batch, sampler=test_sampler, num_workers=arg.num_workers)

	model = Smiles_BERT(len(Smiles_vocab), max_len=arg.seq, nhead=arg.nhead, feature_dim=arg.embed_size, feedforward_dim=arg.model_dim, nlayers=arg.layers, adj=arg.adjacency, dropout_rate=arg.drop_rate)
	model.load_state_dict(torch.load(arg.saved_model))
	output_layer = nn.Linear(arg.embed_size, num_tasks)
	
	model = BERT_base(model, output_layer)
	#model = BERT_base_dropout(model, output_layer)
	
	model.to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	#model.to(device)

	optim = Adam(model.parameters(), lr=arg.lr, weight_decay=0)
	criterion = nn.BCEWithLogitsLoss(reduction='none')
	# load model
	print("Start fine-tuning with seed", arg.seed)
	min_valid_loss = 100000
	counter = 0

	for epoch in range(arg.epoch):
		avg_loss = 0
		valid_avg_loss = 0
		total_hit = 0
		total = 0

		data_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
		#position_num = torch.arange(arg.seq).repeat(arg.batch,1).to(device)
		model.train()
		for i, data in data_iter:
			data = {key:value.to(device) for key, value in data.items()}
			position_num = torch.arange(arg.seq).repeat(data["smiles_bert_input"].size(0),1).to(device)
			if arg.adjacency is True:
				output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
			else:
				output = model.forward(data["smiles_bert_input"], position_num)
			output = output[:,0]
			data["smiles_bert_label"] = data["smiles_bert_label"].view(output.shape).to(torch.float64)
			is_valid = data["smiles_bert_label"] ** 2 > 0

			loss = criterion(output.double(), (data["smiles_bert_label"]+1)/2)
			loss = torch.where(is_valid, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
			optim.zero_grad()
			loss = torch.sum(loss) / torch.sum(is_valid)
			loss.backward()
			#torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
			optim.step()

			avg_loss += loss.item()
			status = {"epoch":epoch, "iter":i, "avg_loss":avg_loss / (i+1), "loss":loss.item()}
			if i % 100 == 0:
				print(i)
				#data_iter.write(str(status))
		print("Epoch: ", epoch, "average loss: ", avg_loss/len(data_iter))

		model.eval()
		valid_iter = tqdm.tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
		#position_num = torch.arange(arg.seq).repeat(arg.batch,1).to(device)
		predicted_list = []
		target_list = []

		with torch.no_grad():
			for i, data in valid_iter:
				data = {key:value.to(device) for key, value in data.items()}
				position_num = torch.arange(arg.seq).repeat(data["smiles_bert_input"].size(0),1).to(device)
				if arg.adjacency is True:
					output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
				else:
					output = model.forward(data["smiles_bert_input"], position_num)
				output = output[:,0]
				data["smiles_bert_label"] = data["smiles_bert_label"].view(output.shape).to(torch.float64)
				is_valid = data["smiles_bert_label"] ** 2 > 0
				valid_loss = criterion(output.double(), (data["smiles_bert_label"]+1)/2)
				valid_loss = torch.where(is_valid, valid_loss, torch.zeros(valid_loss.shape).to(valid_loss.device).to(valid_loss.dtype))
				valid_loss = torch.sum(valid_loss) / torch.sum(is_valid)

				valid_avg_loss += valid_loss.item()
				predicted = torch.sigmoid(output)
				predicted_list.append(predicted)
				target_list.append(data["smiles_bert_label"])
				
				#_, predicted = torch.max(output.data, 1)

				#total += data["smiles_bert_label"].size(0)
				#total_hit += (torch.round(predicted) == data["smiles_bert_label"]).sum().item()
		predicted_list = torch.cat(predicted_list, dim=0).cpu().numpy()
		target_list = torch.cat(target_list, dim=0).cpu().numpy()
		#predicted_list = np.reshape(predicted_list, -1)
		#target_list = np.reshape(target_list, -1)
		roc_list = []
		for i in range(target_list.shape[1]):
			if np.sum(target_list[:,i] == 1) > 0 and np.sum(target_list[:,i] == -1) > 0:
				is_valid = target_list[:,i] ** 2 > 0
				roc_list.append(roc_auc_score((target_list[is_valid,i]+1)/2, predicted_list[is_valid,i]))
		
		print("AUCROC: ", sum(roc_list)/len(roc_list))

		if valid_avg_loss < min_valid_loss:
			save_path = "../finetuned_model/" + str(arg.dataset) + "_epoch_" + str(epoch) + "_val_loss_" + str(round(valid_avg_loss/len(valid_dataloader),5))
			torch.save(model.state_dict(), save_path+'.pt')
			model.to(device)
			min_valid_loss = valid_avg_loss
			counter = 0

		counter += 1
		if counter > 5:
			break

	# eval
	print("Finished. Start evaluation.")
	correct = 0
	total = 0
	predicted_list = []
	target_list = []

	model.eval()
	#test_iter = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
	#position_num = torch.arange(arg.seq).repeat(arg.batch,1).to(device)
	with torch.no_grad():
		for i, data in enumerate(test_dataloader):
			data = {key:value.to(device) for key, value in data.items()}
			position_num = torch.arange(arg.seq).repeat(data["smiles_bert_input"].size(0),1).to(device)
			if arg.adjacency is True:
				output = model(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
			else:
				output = model(data["smiles_bert_input"], position_num)
			output = output[:,0]
			data["smiles_bert_label"] = data["smiles_bert_label"].view(output.shape).to(torch.float64)
			predicted = torch.sigmoid(output)
			predicted_list.append(predicted)
			target_list.append(data["smiles_bert_label"])
			
			#_, predicted = torch.max(output.data, 1)

			#total += data["smiles_bert_label"].size(0)
			#correct += (torch.round(predicted) == data["smiles_bert_label"]).sum().item()
		predicted_list = torch.cat(predicted_list, dim=0).cpu().numpy()
		target_list = torch.cat(target_list, dim=0).cpu().numpy()
		#predicted_list = np.reshape(predicted_list, -1)
		#target_list = np.reshape(target_list, -1)
		roc_list = []
		for i in range(target_list.shape[1]):
			if np.sum(target_list[:,i] == 1) > 0 and np.sum(target_list[:,i] == -1) > 0:
				is_valid = target_list[:,i] ** 2 > 0
				roc_list.append(roc_auc_score((target_list[is_valid,i]+1)/2, predicted_list[is_valid,i]))
		
		print("AUCROC: ", sum(roc_list)/len(roc_list))
	print("Evaluate on min valid loss model")
	correct = 0
	total = 0
	predicted_list = []
	target_list = []
	model.load_state_dict(torch.load(save_path+'.pt'))
	model.eval()
	#test_iter = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
	#position_num = torch.arange(arg.seq).repeat(arg.batch,1).to(device)
	with torch.no_grad():
		for i, data in enumerate(test_dataloader):
			data = {key:value.to(device) for key, value in data.items()}
			position_num = torch.arange(arg.seq).repeat(data["smiles_bert_input"].size(0),1).to(device)
			if arg.adjacency is True:
				output = model(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
			else:
				output = model(data["smiles_bert_input"], position_num)
			output = output[:,0]
			data["smiles_bert_label"] = data["smiles_bert_label"].view(output.shape).to(torch.float64)
			predicted = torch.sigmoid(output)
			predicted_list.append(predicted)
			target_list.append(data["smiles_bert_label"])
			#_, predicted = torch.max(output.data, 1)

			#total += data["smiles_bert_label"].size(0)
			#correct += (torch.round(predicted) == data["smiles_bert_label"]).sum().item()

		#predicted_list = np.reshape(predicted_list, -1)
		#target_list = np.reshape(target_list, -1)
		predicted_list = torch.cat(predicted_list, dim=0).cpu().numpy()
		target_list = torch.cat(target_list, dim=0).cpu().numpy()
		roc_list = []
		for i in range(target_list.shape[1]):
			if np.sum(target_list[:,i] == 1) > 0 and np.sum(target_list[:,i] == -1) > 0:
				is_valid = target_list[:,i] ** 2 > 0
				roc_list.append(roc_auc_score((target_list[is_valid,i]+1)/2, predicted_list[is_valid,i]))
		
		print("AUCROC: ", sum(roc_list)/len(roc_list))

if __name__ == "__main__":
	main()
