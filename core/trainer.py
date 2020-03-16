import torch
import torch.nn as nn


class Trainer:
	def __init__(self, 
		model, 
		train_data,
		teacher_forcing_schedule=None,
		optimizer=None, encoder_optimizer=None, decoder_optimizer=None,
		clip=None,
		on_batch_done_callback=None,
		on_epoch_done_callback=None,):
		try:
			assert(
				(optimizer and not (encoder_optimizer or decoder_optimizer)) 
				or (not optimizer and encoder_optimizer and decoder_optimizer))
		except:
			raise AssertionError('Provide either a single optimizer or both encoder and decoder optimizers.')

		self.model = model
		self.train_data = train_data
		self.loss_function = nn.NLLLoss(ignore_index=0)
		self.teacher_forcing_schedule = teacher_forcing_schedule
		self.optimizer = optimizer
		self.encoder_optimizer = encoder_optimizer if encoder_optimizer else optimizer
		self.decoder_optimizer = decoder_optimizer if encoder_optimizer else optimizer
		self.clip = clip

		self.on_batch_done_callback = on_batch_done_callback
		self.on_epoch_done_callback = on_epoch_done_callback

	def train(self, n_epochs, batch_size):
		total_batches = 0
		for epoch in range(n_epochs):
			self.train_data.reset_with_batch_size(batch_size)
			avg_loss = 0
			n_batches = 0
			for batch in self.train_data:
				teacher_forcing_ratio = self.teacher_forcing_schedule(total_batches)
				batch_loss = self.batch_update(*batch, teacher_forcing_ratio, batch_index=total_batches)
				n_batches += 1
				total_batches += 1
				avg_loss += batch_loss
			avg_loss /= n_batches
			if self.on_epoch_done_callback:
				self.on_epoch_done_callback(epoch, avg_loss.item())

	def batch_update(self, batch_x, batch_y, teacher_forcing_ratio, batch_index):
		if self.optimizer:
			self.optimizer.zero_grad()
		else:
			self.encoder_optimizer.zero_grad()
			self.decoder_optimizer.zero_grad()

		max_length = batch_y.shape[1]
		outputs = self.model(batch_x, max_length, teacher_forcing_ratio, targets=batch_y)
		loss = 0
		for t in range(batch_y.shape[1]):
			try:
				loss += self.loss_function(outputs[:,t,:], batch_y[:,t])
			except:
				import pdb; pdb.set_trace()
		loss.backward()
		if self.clip:
			torch.nn.utils.clip_grad_norm(self.model.parameters, self.clip)
		if self.optimizer:
			self.optimizer.step()
		else:
			self.encoder_optimizer.step()
			self.decoder_optimizer.step()

		if self.on_batch_done_callback:
			self.on_batch_done_callback(batch_index, loss.item(), self.model.parameters)

		return loss
	