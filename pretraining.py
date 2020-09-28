import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam
import numpy as np
import tqdm
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
#from parallel import DataParallelCriterion, DataParallelModel
from optim_scheduler import ScheduledOptim
from data_utils import Vocab, SmilesDataset
from model import Smiles_BERT, Masked_prediction, BERT_base

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help="dataset path", type=str, default = None)
	parser.add_argument('--adjacency', help="use adjacency matrix", type=bool, default=False)
	parser.add_argument('--batch', help="batch size", type=int, default=128)
	parser.add_argument('--epoch', help="epoch", type=int, default=50)
	parser.add_argument('--seq', help="sequence length", type=int, default=256)
	parser.add_argument('--lr', help="learning rate", type=float, default=0.0001)
	parser.add_argument('--embed_size', help="embedding vector size", type=int, default=1024)
	parser.add_argument('--layers', help="number of layers", type=int, default=6)
	parser.add_argument('--nhead', help="number of head", type=int, default=4)
	parser.add_argument('--matrix_position', help="position of adjacency matrix", type=str, default='atom')
	parser.add_argument('--warmup_step', help="warmup step for scheduled learning rate", type=int, default=10000)
	parser.add_argument('--num_workers', help="number of workers", type=int, default=0)
	parser.add_argument("--local_rank", type=int, default=-1)
	parser.add_argument("--seed", type=int, default=7)
	#parser.add_argument('--savepath', help="saved model dir", type=str)
	arg = parser.parse_args()

	torch.manual_seed(arg.seed)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("device:", device)
	Smiles_vocab = Vocab()
	dataset = SmilesDataset(arg.path, Smiles_vocab, seq_len=arg.seq, mat_position=arg.matrix_position)
	print("Dataset loaded")
	#indices = list(range(len(dataset)))
	#split = 60000
	#np.random.seed(random_seed)
	#np.random.shuffle(indices)

	#train_idx, valid_idx = indices[split:], indices[:split]
	#train_sampler = SubsetRandomSampler(train_idx)
	#valid_sampler = SubsetRandomSampler(valid_idx)

	train_dataloader = DataLoader(dataset, shuffle=True, batch_size=arg.batch, num_workers=arg.num_workers, pin_memory=True)
	#valid_dataloader = DataLoader(dataset, batch_size=arg.batch, sampler=valid_sampler, num_workers=4)
	

	# use adj matrix or not
	model = Smiles_BERT(len(Smiles_vocab), max_len=arg.seq, nhead=arg.nhead, model_dim=arg.embed_size, nlayers=arg.layers, adj=arg.adjacency)
	output_layer = Masked_prediction(arg.embed_size, len(Smiles_vocab))
	model = BERT_base(model, output_layer)
	model.to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		#model = DataParallelModel(model)

	optim = Adam(model.parameters(), lr=arg.lr, weight_decay=0)
	scheduled_optim = ScheduledOptim(optim, arg.embed_size, n_warmup_steps=arg.warmup_step)

	# do not learn labeled as 0 (padding + not masked)
	criterion = nn.CrossEntropyLoss(ignore_index=0)
	#criterion = nn.NLLLoss(ignore_index=0)
	#criterion = DataParallelCriterion(criterion) 

	print("Start pre-training")
	for epoch in range(arg.epoch):
		avg_loss = 0
		#hit = 0
		#total = 0
		data_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
		position_num = torch.arange(arg.seq).repeat(arg.batch,1).to(device)
		model.train()
		for i, data in data_iter:
			data = {key:value.to(device) for key, value in data.items()}
			if data["smiles_bert_input"].size(0) != arg.batch:
				position_num = torch.arange(arg.seq).repeat(data["smiles_bert_input"].size(0),1).to(device)
			if arg.adjacency is True:
				output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
			else:
				output = model.forward(data["smiles_bert_input"], position_num)
			#print(output.shape, data["smiles_bert_label"].shape)
			#print(output, data["smiles_bert_label"])
			loss = criterion(output.transpose(1,2), data["smiles_bert_label"])
			scheduled_optim.zero_grad()
			loss.backward()
			#torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
			scheduled_optim.step_and_update_lr()

			avg_loss += loss.item()

			status = {"epoch":epoch, "iter":i, "avg_loss":avg_loss / (i+1), "loss":loss.item()}
			if i % 100 == 0:
				data_iter.write(str(status))
			if i % 5000 == 0:
				#print()
				torch.save(model.module.state_dict(), "../saved_model/temp_model_" + "epoch_" + str(epoch) + "_" + str(i) + "_" + str(round(avg_loss / (i+1),5)))
			#hit = output.argmax(dim=-1).eq(data["smiles_bert_label"])

		print("Epoch: ", epoch, "average loss: ", avg_loss/len(data_iter))

		save_path = "../saved_model/" + "nlayers_"+ str(arg.layers) + "_nhead_" + str(arg.nhead) + "_adj_" + str(arg.adjacency) + "_epoch_" + str(epoch) + "_loss_" + str(round(avg_loss/len(data_iter),5))
		torch.save(model.module.bert.state_dict(), save_path+'.pt')
		model.to(device)
		print("model saved")


if __name__ == "__main__":
	main()