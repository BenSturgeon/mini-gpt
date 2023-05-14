# Mini-GPT: an introductory level implementation of a GPT

This project is an implementation of a simplified version of the Generative Pre-training Transformer (GPT), named Mini-GPT. This is intended for educational purposes to help understand the basics of GPTs and Transformer models. While it does not contain all the bells and whistles of the full GPT models developed by OpenAI (like GPT-3 or GPT-4), it provides an excellent starting point for understanding these powerful natural language processing tools. This project is inspired by the nano-GPT project by Andrej Karpathy, but I wanted to work through the code on my own and write it in a way which to me is more clear.

Table of Contents
1. Introduction
2. Requirements
3. Installation
4. Usage
5. Contribute
6. License

## Introduction
The aim of this project is to implement the basic structures that train and deploy a generative pretrained transformer (or GPT). The goal is to produce text that vaguely resembles the original source material, and offer some insight into how scaling can radically influence the performance of these models.

I aim to explore and clarify some of the key concepts that go into the GPT by dealing with the material in a hands-on way, but with extra descriptions of my understanding of the concepts to try build an intuition of why the code works. This should be a helpful resource to beginners approaching NLP and AI.

I highly recommend checking out the original video this project is based on [here](https://www.youtube.com/watch?v=kCc8FmEb1nY&t) with Andrej Karpathy.

## Requirements
Python 3.7 or later
PyTorch 1.4 or later



## Installation
Clone the repository and install the necessary packages:



git clone https://github.com/bensturgeon/mini-gpt.git
cd mini-gpt
pip install -r requirements.txt

## Usage
To train the Mini-GPT model, you can use the following command:

python train.py --data_path /path/to/your/data_as_a_single_file
After training, you can generate text using the following command:

python generate.py --model_path /path/to/your/model

## Contribute
While I don't expect any contributions, if you wish to do so please don't hesitate to make a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Disclaimer
This project is not affiliated, associated, authorized, endorsed by, or in any way officially connected with OpenAI or any of its subsidiaries or its affiliates. The official OpenAI website can be found at https://www.openai.com. The names GPT, GPT-3, GPT-4, OpenAI, as well as related names, marks, emblems, and images are registered trademarks of their respective owners.

I give credit to the structure and ideas here both to the authors of the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper and [Andrej Karpathy](https://github.com/karpathy/nanoGPT)

## Contact
If you have any questions, feel free to open an issue or contact me directly.