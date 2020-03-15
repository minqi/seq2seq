import unicodedata
import re

import numpy as np
import torch
from torch.autograd import Variable

import utils.log as log


SOS_TOKEN = 1
EOS_TOKEN = 0

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.n_words = 2

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class ParallelCorpus:
    def __init__(self, path, lang1, lang2, reverse=False, _filter=None, verbose=True, device='cpu'):
        self.input_lang = lang1
        self.output_lang = lang2
        self.pairs = []
        self.device = device
        self.shuffled_indices = []
        self.batch_size = 1
        self.batches_generated = 0
        if path:
            input_lang, output_lang, pairs = \
                self.load_parallel_corpus(path, lang1, lang2, 
                    reverse=reverse, _filter=_filter, verbose=verbose)

    def _unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def _normalize_string(self, s):
        s = self._unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def load_parallel_corpus(self, path, lang1, lang2, reverse=False, _filter=None, verbose=True):
        log.cprint(verbose, 'Loading language pairs from %s...' % (path))

        lines = open(path).read().strip().split('\n')
        pairs = [[self._normalize_string(s) for s in l.split('\t')[:2]] for l in lines]

        log.cprint(verbose, 'Loaded %i sentence pairs' % len(pairs))

        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Language(lang2)
            output_lang = Language(lang1)
        else:
            input_lang = Language(lang1)
            output_lang = Language(lang2)

        if _filter:
            pairs = [p for p in pairs if _filter(p)]
            log.cprint(verbose, 'Filtered to %i sentence pairs' % len(pairs))

        log.cprint(verbose, 'Indexing words...')
        for p in pairs:
            input_lang.index_words(p[0])
            output_lang.index_words(p[1])

        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.shuffle_dataset()

        return input_lang, output_lang, pairs

    def indices_from_sentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensor_from_sentence(self, lang, sentence):
        indices = self.indices_from_sentence(lang, sentence)
        indices.append(EOS_TOKEN)
        return Variable(torch.LongTensor(indices)).to(self.device)

    def tensors_from_pair(self, pair):
        input_tensor = self.tensor_from_sentence(self.input_lang, pair[0])
        output_tensor = self.tensor_from_sentence(self.output_lang, pair[1])
        return (input_tensor, output_tensor)

    def get_random_batch(self, batch_size=1, replace=False):
        indices = np.random.choice(range(len(self.pairs)), batch_size, replace=replace)
        batch_pairs = [self.tensors_from_pair(self.pairs[i]) for i in indices]
        batch_in = torch.nn.utils.rnn.pad_sequence([p[0] for p in batch_pairs], batch_first=True)
        batch_out = torch.nn.utils.rnn.pad_sequence([p[1] for p in batch_pairs], batch_first=True)
        return (batch_in, batch_out)

    @property
    def corpus_size(self):
        return len(self.pairs)

    def sample_corpus(self):
        start = self.batch_size*self.batches_generated
        batch_pairs = [
            self.tensors_from_pair(self.pairs[i])
            for i in range(start, start + self.batch_size)
            if i < self.corpus_size]
        batch_in = torch.nn.utils.rnn.pad_sequence([p[0] for p in batch_pairs], batch_first=True)
        batch_out = torch.nn.utils.rnn.pad_sequence([p[1] for p in batch_pairs], batch_first=True)
        return (batch_in, batch_out)

    def shuffle_dataset(self):
        self.shuffled_indices = torch.randperm(len(self.pairs))

    def reset_with_batch_size(self, batch_size=1):
        self.batch_size = batch_size
        self.batches_generated = 0
        self.shuffle_dataset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated*self.batch_size >= self.corpus_size:
            self.shuffle_dataset()
            raise StopIteration()

        next_batch = self.sample_corpus()
        self.batches_generated += 1

        return next_batch
        