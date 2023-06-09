
from bigram_model import BigramLanguageModel, show_output
import argparse
import sys
import torch
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import urllib.request

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='Path to a model file to load')
    parser.add_argument('--save', type=str, default=None, help='Path to save the trained model')
    parser.add_argument('--train_time', type=int, default=1000, help='Training iterations')
    parser.add_argument('--output_path', type=str, default='log.txt', help='Path to save log')
    parser.add_argument('--num_tokens', type=int, default=250, help='number of output tokens')
    parser.add_argument('--reverse_keys', type=bool, default=False, help='test reversal of keys and query networks')



    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = BigramLanguageModel()
    model = model.to(device)
    if args.load != None:
        load_path = args.load
    else:
        load_path = "fire-gpt.pth"
    if args.output_path != None:
        output_path = args.output_path
    else:
        output_path = "output_text.txt"

    
    if args.load is not None:
        print("Hey ben")
        model.load_state_dict(torch.load(load_path))
        print(model.load_state_dict(torch.load(load_path)))
        print(output_path)

        show_output(model, output_path, num_tokens=args.num_tokens)
    else: 
        print("please add a directory from which to load the model.")


if __name__ == "__main__":
    main()