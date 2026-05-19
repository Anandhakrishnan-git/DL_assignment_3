# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## W&B Report

[View the full experiment report on Weights & Biases](https://api.wandb.ai/links/anandhakrishnanm21-indian-institute-of-technology-madras/gyn7jek2)

## GitHub Repository

[https://github.com/Anandhakrishnan-git/DL_assignment_3](https://github.com/Anandhakrishnan-git/DL_assignment_3.git)

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```

