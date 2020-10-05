import torch
import torch.nn as nn
from Embedding import Smiles_embedding

class BERT_double_tasks(nn.Module):
	def __init__(self, model, out1, out2):
		super().__init__()
		self.bert = model
		self.linear_value = out1
		self.linear_mask = out2
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		return self.linear_value(x[:,0]), self.linear_mask(x)

class BERT_add_feature(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.bert = model
		self.linear = output_layer
	def forward(self, x, feature, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(torch.cat((x[:,0], feature), dim=1))
		return x

class BERT_base(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.bert = model
		self.linear = output_layer
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(x)
		return x

class BERT_base_dropout(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.bert = model
		self.linear = output_layer
		self.drop = nn.Dropout(0.2)
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(x)
		x = self.drop(x)
		return x

class Smiles_BERT_BC(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.smiles_bert = model
		self.linear = output_layer
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.smiles_bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(torch.mean(x, 1))
		return x

class classification_layer(nn.Module):
	def __init__(self, hidden):
		super().__init__()
		self.linear = nn.Linear(hidden,1)
	def forward(self, x):
		return self.linear(x)

class Masked_prediction(nn.Module):
	def __init__(self, hidden, vocab_size):
		super().__init__()
		self.linear = nn.Linear(hidden, vocab_size)
	def forward(self, x):
		return self.linear(x)

class Smiles_BERT(nn.Module):
	def __init__(self, vocab_size, max_len=256, feature_dim=1024, nhead=4, feedforward_dim=1024, nlayers=6, adj=False, dropout_rate=0):
		super(Smiles_BERT, self).__init__()
		self.embedding = Smiles_embedding(vocab_size, feature_dim, max_len, adj=adj)
		trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu', dropout=dropout_rate)
		self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)
		
		#self.linear = Masked_prediction(feedforward_dim, vocab_size)

	def forward(self, src, pos_num, adj_mask=None, adj_mat=None):
		# True -> masking on zero-padding. False -> do nothing
		#mask = (src == 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
		mask = (src == 0)
		mask = mask.type(torch.bool)
		#print(mask.shape)

		x = self.embedding(src, pos_num, adj_mask, adj_mat)
		x = self.transformer_encoder(x.transpose(1,0), src_key_padding_mask=mask)
		x = x.transpose(1,0)
		#x = self.linear(x)
		return x