import random

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def show_plot(points):
	plt.figure()
	fig, ax = plt.subplots()
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
	plt.show()

def evaluate(model, corpus, sentence=None, max_length=10, render_attention=True):
	pair = None
	if not sentence:
		pair = random.choice(corpus.pairs)
		sentence = pair[0]

	with torch.no_grad():
		prediction, attention = model.predict(sentence, corpus, max_length)
		print('>', sentence)
		if not sentence:
			print('=', pair[1])
		print('<', prediction)

		if render_attention:
			show_attention(sentence, prediction, attention)

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()