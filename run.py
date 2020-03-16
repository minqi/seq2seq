import argparse, os, time
from datetime import datetime
from functools import partial

import torch

from utils.language import ParallelCorpus
from models.seq2seq import EncoderRNN, AttentionDecoderRNN, BahdanauAttentionDecoderRNN, Seq2Seq 
from core.trainer import Trainer
import utils.log as log
import utils.evaluation as evaluation


def load_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--data_path', type=str, default='data/translation/en-fr-small.txt')
	parser.add_argument('-i', '--input_name', type=str, default='en')
	parser.add_argument('-o', '--output_name', type=str, default='fr')
	parser.add_argument('-s', '--save_path', type=str, default='checkpoints/')
	parser.add_argument('-v', '--verbose', type=bool, default=True)
	parser.add_argument('--cuda', type=bool, default=False)

	parser.add_argument('--hidden_size', type=int, default=256)
	parser.add_argument('--dropout', type=float, default=0.05)

	parser.add_argument('-bs', '--batch_size', type=int, default=1)
	parser.add_argument('-lr', '--learning_rate', type=int, default=1e-4)
	parser.add_argument('-n', '--n_epochs', type=int, default=50000)
	parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
	parser.add_argument('--clip', type=float, default=5.0)

	parser.add_argument('--checkpoint_epochs', type=int, default=10)
	parser.add_argument('--print_epochs', type=int, default=10)
	parser.add_argument('--plot_epochs', type=int, default=10)

	return parser.parse_args()

# For faster results
MAX_LENGTH = 10
GOOD_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
)
def simple_filter(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
		p[0].startswith(GOOD_PREFIXES)

def log_epoch(n_epochs, checkpoint_epochs, print_epochs, plot_epochs, 
		save_path, plot_losses, since, epoch, loss, model, optimizer_dict):
	if epoch == 0:
		return

	if epoch % print_epochs == 0:
		time_stats = log.time_since(start, epoch / n_epochs)
		print(f"{'epoch: ' + str(epoch):15}{'loss: ' + str(loss):15}{time_stats:15}")

	if epoch % plot_epochs == 0:
		plot_losses.append(loss)

	if epoch % checkpoint_epochs == 0:
		torch.save({
			'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_dict,
            'loss': loss}, save_path)

if __name__ == '__main__':
	args = load_args()

	device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

	corpus = ParallelCorpus(
				args.data_path, args.input_name, args.output_name,
				_filter=simple_filter, verbose=args.verbose, device=device)

	encoder = EncoderRNN(
		corpus.input_lang.n_words, args.hidden_size, n_layers=2).to(device)
	decoder = AttentionDecoderRNN(
		'general', args.hidden_size, corpus.output_lang.n_words, n_layers=2, dropout=args.dropout).to(device)
	seq2seq = Seq2Seq(encoder, decoder).to(device)

	# teacher_forcing_schedule = lambda i:args.teacher_forcing_ratio 
	# optimizer = torch.optim.Adam(seq2seq.parameters, lr=args.learning_rate)

	# plot_losses = []
	# save_path = os.path.join(args.save_path, 's2s-' + datetime.now().strftime('%d-%m-%Y-%h-%S.%f'))
	# start = time.time()
	# on_epoch_done_callback = partial(log_epoch,
	# 	args.n_epochs, args.checkpoint_epochs, args.print_epochs, args.plot_epochs, 
	# 	save_path, plot_losses, start)
	# trainer = Trainer(seq2seq, corpus, teacher_forcing_schedule, optimizer=optimizer, clip=args.clip,
	# 		on_epoch_done_callback=on_epoch_done_callback)

	# trainer.train(args.n_epochs, args.batch_size)

	# evaluation.show_plot(plot_losses)
 
	checkpoint = torch.load('checkpoints/s2s-16-03-2020-Mar-27.393182')
	seq2seq.load_state_dict(checkpoint['model_state_dict'])
	seq2seq.eval()
	sentence = "i m ok ."
	evaluation.evaluate(seq2seq, corpus, sentence)

