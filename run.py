import argparse

import torch

from utils.language import ParallelCorpus
from models.seq2seq import EncoderRNN, AttentionDecoderRNN, BahdanauAttentionDecoderRNN

def load_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--data_path', type=str, default='data/translation/en-fr-small.txt')
	parser.add_argument('-i', '--input_name', type=str, default='en')
	parser.add_argument('-o', '--output_name', type=str, default='fr')
	parser.add_argument('-v', '--verbose', type=bool, default=True)
	parser.add_argument('--cuda', type=bool, default=False)

	return parser.parse_args()

# For faster results
MAX_LENGTH = 10
GOOD_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "hello", "hi",
)
def simple_filter(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
		p[0].startswith(GOOD_PREFIXES)

if __name__ == '__main__':
	args = load_args()

	device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

	corpus = ParallelCorpus(
				args.data_path, args.input_name, args.output_name,
				_filter=simple_filter, verbose=args.verbose, device=device)

	corpus.reset_with_batch_size(2)
	for b in corpus:
		x, y, lengths = b
		print(x)
		print(y)
		print(lengths)

	encoder = EncoderRNN(10, 16, 2).to(device)
	# decoder = AttentionDecoderRNN('dot', 16, 10, 2)
	decoder = BahdanauAttentionDecoderRNN(16, 10, 2)
	print(encoder)
	print(decoder)

	word_inputs = torch.LongTensor([[1, 2, 3], [1, 2, 0]])

	bs = word_inputs.shape[0]
	encoder_out, encoder_h, encoder_lengths  = encoder(word_inputs, None, reset=True)
	decoder_h = encoder_h
	last_context = torch.zeros(bs, decoder.hidden_size)

	decoder_outs = []
	for i in range(encoder_out.shape[1]):
		# decoder_out, decoder_h, decoder_c, decoder_a = \
		# 	decoder(word_inputs[:, i], last_context, decoder_h, encoder_out, encoder_lengths)
		decoder_out, decoder_h, decoder_a = \
			decoder(word_inputs[:, i], decoder_h, encoder_out, encoder_lengths)
		decoder_outs.append(decoder_out)

	decoder_outs = torch.stack(decoder_outs, 1)
