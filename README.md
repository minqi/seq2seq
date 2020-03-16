# seq2seq

A seq2seq implementation with support for the following:
- Batch updates
- Bahdanau and Luong attention-based decoding
- Teacher forcing via scheduled sampling

Example translation data can be downloaded [here](http://www.manythings.org/anki/).

This codebase started as a modular reimplementation of [this seq2seq tutorial](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb).

### Get started
```
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```
### References
- Bahdanau et al. Neural Machine Translation by Jointly Learning to Align and Translate. 2014. [link](https://arxiv.org/abs/1409.0473)
- Luong et al. Effective Approaches to Attention-based Neural Machine Translation. 2015. [link](https://arxiv.org/abs/1508.04025)
- Bengio et al. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. 2015. [link](https://arxiv.org/abs/1506.03099)
