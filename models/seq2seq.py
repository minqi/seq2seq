import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.sequence_utils as sequence_utils
from utils import language


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

	def forward(self, word_inputs, hidden=None, reset=True):
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
		self.out = nn.Linear(2*hidden_size, output_size) 
		# @todo: use max-out with k=2 followed by linear deep output

	def forward(self, word_input, last_context, last_hidden, encoder_outputs, encoder_output_lengths):
		# N.B. Bahdanau-style attention decoding does not use the last context, but this argument
		# is included here for consistency with the more recent Luong-style decoder.
		word_embedding = self.dropout(self.embedding(word_input))

		# Use h_{t-1} to predict attention scores. Hidden encodes previous context, so no need for "input feeding."
		mask = sequence_utils.length_to_mask(encoder_output_lengths)
		attention_scores = self.attention_layer(last_hidden[-1], encoder_outputs, mask)
		context = attention_scores.unsqueeze(1).bmm(encoder_outputs).squeeze(1)

		rnn_input = torch.cat((word_embedding, context), -1).unsqueeze(1)
		rnn_output, hidden = self.gru(rnn_input, last_hidden)
		rnn_output = rnn_output.squeeze(1)
		output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), 1)

		return output, hidden, context, attention_scores


class Seq2Seq(DeviceAwareModule):
	def __init__(self, encoder, decoder):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder

	@property
	def parameters(self):
		return list(self.encoder.parameters()) + list(self.decoder.parameters())

	@property
	def encoder_parameters(self):
		return self.encoder.parameters()

	@property
	def decoder_parameters(self):
		return self.decoder.parameters()

	def forward(self, word_inputs, max_length, teacher_forcing_ratio=0.0, targets=None):
		bs = word_inputs.shape[0]
		encoder_out, encoder_h, encoder_lengths = self.encoder(word_inputs)
		
		decoder_h = encoder_h
		decoder_c = torch.zeros(bs, self.decoder.hidden_size)
		decoder_input = \
			torch.LongTensor([language.SOS_TOKEN]).repeat(bs).to(self.decoder.device)

		done = torch.BoolTensor(bs).fill_(False)
		outputs = []
		for t in range(max_length):
			decoder_out, decoder_h, decoder_c, decoder_a = \
				self.decoder(decoder_input, decoder_c, decoder_h, encoder_out, encoder_lengths)

			if self.training:
				outputs.append(decoder_out)
			else:
				max_out = decoder_out.argmax(1)
				max_out[done] = language.PAD_TOKEN
				outputs.append(max_out)
				done ^= (max_out == language.EOS_TOKEN)
				if done.all():
					break

			use_teacher_forcing = self.training and random.random() < teacher_forcing_ratio
			if use_teacher_forcing:
				decoder_input = targets[:, t]
			else:
				decoder_input = decoder_out.argmax(1)

		outputs = torch.stack(outputs, 1) # B x T x D

		return outputs

	def predict(self, sentence, corpus, max_length=10):
		word_inputs = corpus.input_lang.tensor_from_sentence(sentence).unsqueeze(0)

		if len(word_inputs.shape) < 2:
			word_inputs = word_inputs.unsqueeze(0)
		encoder_out, encoder_h, encoder_lengths = self.encoder(word_inputs)
		decoder_h = encoder_h
		decoder_c = torch.zeros(1, self.decoder.hidden_size)
		decoder_input = \
			torch.LongTensor([language.SOS_TOKEN]).to(self.decoder.device)
		decoded_words = []
		decoder_attention = torch.zeros(max_length, max_length)
		input_length = word_inputs.shape[1]
		for t in range(max_length):
			decoder_out, decoder_h, decoder_c, decoder_a = \
				self.decoder(decoder_input, decoder_c, decoder_h, encoder_out, encoder_lengths)

			decoder_attention[t, :input_length] += decoder_a[0]

			word_index = decoder_out.argmax(1)
			decoded_words.append(corpus.output_lang.index2word[word_index[0].item()])
			if word_index == language.EOS_TOKEN:
				break

			decoder_input = word_index
			
		return decoded_words, decoder_attention[:t+1, :input_length]


