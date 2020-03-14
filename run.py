import argparse

import torch

from utils.language import ParallelCorpus


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
    "you are", "you re "
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

	# print(corpus.pairs)
	# print(corpus.get_random_batch(2))

	corpus.reset_with_batch_size(2)
	for b in corpus:
		print(b)
