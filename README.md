# Traducteur
### Introduction

This project implements a Transformer-based Neural Machine Translation (NMT) model from scratch using PyTorch. The model is trained to translate French sentences into English and evaluated using the BLEU score metric.

The implementation includes custom modules for attention, positional encoding, vocabulary handling, and beam search-based inference.

### Features

 - Full Transformer architecture implemented from scratch
 - Custom vocabulary and tokenization pipeline 
 - Beam search decoding for inference
 - BLEU score evaluation using NLTK Modular and extensible PyTorch design

### Architecture
<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="Architecture of Transformer" width="500" />

### Key Components

 - Multi-head self-attention
 -  Encoder-decoder architecture 
 - Positional encoding 
 - Masking (padding + causal) 
 - Inference Pipeline
 - Beam search decoding
 - Custom batching and padding
 - BLEU score computation 


### Installation
#### Requirements
	Python 3.8+
	PyTorch
	pandas
	nltk
	tqdm
#### Install dependencies
	pip install torch pandas nltk tqdm
### Usage
#### Evaluate Model (BLEU Score)
	python inference.py

This loads the trained checkpoint and computes the BLEU score on the test dataset.

#### Translate a Sentence

Uncomment the input section in inference.py:

	src = input("Enter sentence to translate: ")
	outputs = translate(model, src)
	print(outputs)
### Model Configuration
#### Parameter	Value
	d_model	128
	num_heads	4
	num_layers	2
	d_ff	512
	max_seq_len	55
	decoding length	10

Text is normalized using Unicode normalization and punctuation removal.

### Evaluation
Metric: BLEU score (corpus-level)
Library: NLTK
Result, BLEU Score ~ 0.32


 ### Reference
Vaswani et al., 2017
*Attention Is All You Need*
https://arxiv.org/abs/1706.03762
