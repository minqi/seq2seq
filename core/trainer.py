import torch


def train(batch_x, batch_y, encoder, decoder, 
	encoder_optimizer, decoder_optimizer, loss_function, max_length=10):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0

	# @todo: need to mask out PAD tokens in batch when computing loss