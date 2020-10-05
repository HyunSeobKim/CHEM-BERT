import torch.nn as nn
import torch
import math

class Smiles_embedding(nn.Module):
	def __init__(self, vocab_size, embed_size, max_len, adj=False):
		super().__init__()
		self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.position = nn.Embedding(max_len, embed_size)
		self.max_len = max_len
		self.embed_size = embed_size
		if adj:
			self.adj = Adjacency_embedding(max_len, embed_size)

		self.embed_size = embed_size

	def forward(self, sequence, pos_num, adj_mask=None, adj_mat=None):
		x = self.token(sequence) + self.position(pos_num)
		if adj_mat is not None:
			# additional embedding matrix. need to modify
			#print(adj_mask.shape)
			x += adj_mask.unsqueeze(2) * self.adj(adj_mat).repeat(1, self.max_len).reshape(-1,self.max_len, self.embed_size)
		return x

class Adjacency_embedding(nn.Module):
	def __init__(self, input_dim, model_dim, bias=True):
		super(Adjacency_embedding, self).__init__()

		self.weight_h = nn.Parameter(torch.Tensor(input_dim, model_dim))
		self.weight_a = nn.Parameter(torch.Tensor(input_dim))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(model_dim))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight_h.size(1))
		stdv2 = 1. /math.sqrt(self.weight_a.size(0))
		self.weight_h.data.uniform_(-stdv, stdv)
		self.weight_a.data.uniform_(-stdv2, stdv2)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input_mat):
		a_w = torch.matmul(input_mat, self.weight_h)
		out = torch.matmul(a_w.transpose(1,2), self.weight_a)

		if self.bias is not None:
			out += self.bias
		#print(out.shape)
		return out
