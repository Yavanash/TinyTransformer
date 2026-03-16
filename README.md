# TinyTransformer

TinyTransformer is a from-scratch PyTorch implementation of the **Transformer architecture** introduced in *Attention Is All You Need* (Vaswani et al., 2017).

The goal of this project is to **incrementally build the full Transformer architecture from first principles**, focusing on understanding every component rather than relying on high-level libraries.

## Objectives

* Implement the Transformer architecture step-by-step
* Understand the mechanics of self-attention and multi-head attention
* Build an encoder-decoder model for **Neural Machine Translation** from French to English
* Maintain clean, minimal, research-style code

## Architecture

The final model follows the architecture proposed in *Attention Is All You Need*:

* Token Embeddings
* Positional Encoding
* Multi-Head Self Attention
* Position-wise Feed Forward Network
* Residual Connections + Layer Normalization
* Encoder Stack
* Decoder Stack with Masked Self-Attention
* Encoder-Decoder Attention
* Linear Projection + Softmax

<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="Architecture of Transformer" width="1000" />


## Goal

The objective is not just to reproduce results, but to **deeply understand how Transformers work internally** by implementing every component manually.

## Reference

Vaswani et al., 2017
*Attention Is All You Need*
https://arxiv.org/abs/1706.03762
