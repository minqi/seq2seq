import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.sequence_utils as sequence_utils


class DeviceAwareModule(nn.Module):
	@property
	def device(self):
		return next(self.parameters()).device
	

class EncoderRNN(DeviceAwareModule):
	def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
			batch_first=True, bidirectional=bidirectional)

	def init_hidden(self, batch_size=1):
		hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
		return hidden

	def forward(self, word_inputs, hidden=None, reset=False):
		bs = word_inputs.shape[0]
		
		if hidden is None or reset:
			hidden = self.init_hidden(batch_size=bs)
		
		input_lengths = (word_inputs != 0).sum(1)
		embeddings = self.embedding(word_inputs)
		embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, batch_first=True)
		output, hidden = self.gru(embeddings, hidden)
		output, lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		return output, hidden, lengths


class Attention(DeviceAwareModule):
	# @todo: need to mask out PAD tokens in batch
	def __init__(self, attention_type, hidden_size):
		super().__init__()

		self.attention_type = attention_type
		self.hidden_size = hidden_size
		
		if self.attention_type == 'concat':
			self.attention_v = nn.Parameter(torch.ones(1, hidden_size).float())
			self.attention = nn.Linear(2*self.hidden_size, hidden_size)
		else:
			self.attention = nn.Linear(self.hidden_size, hidden_size)

	def forward(self, hidden, encoder_outputs, mask=None):
		seq_len = encoder_outputs.shape[1]
		bs = encoder_outputs.shape[0]
		attention_scores = torch.zeros(bs, seq_len).to(self.device) # B x S
		for t in range(seq_len):
			attention_scores[:, t] = self.score(hidden, encoder_outputs[:, t])

		if mask is not None:
			attention_scores *= mask
		return F.softmax(attention_scores, dim=1)

	def score(self, hidden, encoder_output):
		# handle batch logic better here using torch.bmm
		if self.attention_type == 'dot':
			score = (hidden*encoder_output).sum(1)
		elif self.attention_type == 'general':
			score = (self.attention(encoder_output)*hidden).sum(1)
		elif self.attention_type == 'concat':
			score = (torch.tanh(self.attention(torch.cat((hidden, encoder_output), 1))) \
					@ self.attention_v.t()).flatten()
		else:
			raise ValueError('Invalid score function')

		return score


class AttentionDecoderRNN(DeviceAwareModule):
	def __init__(self, attention_type, hidden_size, output_size, n_layers=1, dropout=0.1):
		super().__init__()

		self.attention_type = attention_type
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(2*hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
		self.out = nn.Linear(2*hidden_size, output_size)

		if attention_type:
			self.attention_layer = Attention(attention_type, hidden_size)

	def forward(self, word_input, last_context, last_hidden, encoder_outputs, encoder_output_lengths):
		word_embedding = self.embedding(word_input)
		rnn_input = torch.cat((word_embedding, last_context), -1).unsqueeze(1) # "input feeding" previous alignment context
		rnn_output, hidden = self.gru(rnn_input, last_hidden)
		rnn_output = rnn_output.squeeze(1)

		# Use current h_t to predict attention scores
		mask = sequence_utils.length_to_mask(encoder_output_lengths)
		attention_scores = self.attention_layer(rnn_output, encoder_outputs, mask)
		context = attention_scores.unsqueeze(1).bmm(encoder_outputs).squeeze(1)
		output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1)

		return output, hidden, context, attention_scores


class BahdanauAttentionDecoderRNN(DeviceAwareModule):
	def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
		super().__init__()

		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.attention_layer = Attention('concat', hidden_size)
		self.gru = nn.GRU(2*hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
		self.out = nn.Linear(hidden_size, output_size) 
		# @todo: use max-out with k=2 followed by linear deep output

	def forward(self, word_input, last_hidden, encoder_outputs, encoder_output_lengths):
		word_embedding = self.dropout(self.embedding(word_input))

		# Use h_{t-1} to predict attention scores. Hidden encodes previous context, so no need for "input feeding."
		mask = sequence_utils.length_to_mask(encoder_output_lengths)
		attention_scores = self.attention_layer(last_hidden[-1], encoder_outputs, mask)
		context = attention_scores.unsqueeze(1).bmm(encoder_outputs)

		rnn_input = torch.cat((word_embedding.unsqueeze(1), context), -1)
		rnn_output, hidden = self.gru(rnn_input, last_hidden)
		output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), 1)

		return output, hidden, attention_scores




