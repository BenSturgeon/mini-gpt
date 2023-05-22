import torch
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import urllib.request

# define variables


batch_size = 32
block_size = 128
n_embd = 192
n_head =4
n_layer = 4
lr = 2e-3
dropout = 0.2
training_iters = 5000
eval_interval = 300
eval_iters = 200
output_length = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)



# download tiny shakespeare
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# download the file directly to a variable
text = urllib.request.urlopen(url).read().decode('utf-8')

tokens = list(set(text))
vocab_size = len(tokens)

# Create an encoder decoder for our tokens to turn them into numbers and back
encoder_decoder = {token: i for i, token in enumerate(tokens)}
decoder_encoder = {i: token for i, token in enumerate(tokens)}

encode = lambda x: [encoder_decoder[i] for i in x]
decode = lambda x: "".join([decoder_encoder[i] for i in x])


data = torch.tensor(encode(text), dtype=torch.long)

# Split the data into training and validation sets
split_val = int(len(data) * 0.9)
train_data = data[:split_val]
val_data = data[split_val:]



def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    batch_start_indexes = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in batch_start_indexes])
    y = torch.stack([data[i+1:i+block_size+1] for i in batch_start_indexes])
    
    x,y = x.to(device), y.to(device)

    return x,y

@torch.no_grad() # tells pytorch we don't intend to do backprop. saves memory by not saving gradients.
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
        

class Head(nn.Module):
    """A single self-attention head"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,  bias=False)
        self.query = nn.Linear(n_embd,head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)      
        q = self.query(x)   

        # To determine the attention of words (more exactly tokens) we use ‘queries’, ‘keys’ and ‘values’.
        # All of them are presented in vectors. 
        # Keys activate depending on the strength of closeness with the query vector as determined by dot product.
        # Keys are an encoded representation for values, in simple cases they can be the same. 
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,16) @ (B,16,T) ---> B, T, T: our desired shape

        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # Projection back into the residual pathway
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, self.head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x +self.sa(self.ln1(x)) # The adding of the values to x is our residual connections, or skip connections
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x= tok_emb + pos_emb
        x= self.blocks(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        
        else:
            # Where 
            # B = batch_size = 4
            # T = time = 8
            # C = channel = 65 = vocab_size
            #  We change the shapes of our logits to get them in the shape needed to use pytorch's cross_entropy function

            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
        return logits
    
    def generate(self, x_input, max_new_tokens):

        for _ in range(max_new_tokens):
            reduced_x_input = x_input[:,-block_size:]
            logits, loss = self(reduced_x_input) # we're not using loss, as we're generating

            next_token = logits[:, -1,:]

            probabilities = F.softmax(next_token, dim=-1)

            top_answer = torch.multinomial(probabilities, num_samples=1)

            x_input = torch.cat((x_input, top_answer), dim=1) # B, T+1. Appending to 1st dimension which is the time dimension

        return x_input

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))


import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='Path to a model file to load')
    parser.add_argument('--save', type=str, default=None, help='Path to save the trained model')
    parser.add_argument('--train_time', type=int, default=1000, help='Training iterations')
    parser.add_argument('--output_path', type=str, default='log.txt', help='Path to save log')

    args = parser.parse_args()

    model = BigramLanguageModel()
    model = model.to(device)
    
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    with open(args.output_path, 'w') as f:
        f.write(f'Model is on device: {next(model.parameters()).device}\n')

    for iter in range(args.train_time):
        if iter % eval_interval == 0:
            averaged_losses = estimate_loss(model)
            with open(args.output_path, 'a') as f:
                f.write(f"steps: {iter}  train loss:{averaged_losses['train']:.4f}  test loss:{averaged_losses['val']:.4f}\n")

        xb, yb = get_batch(batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if args.save is not None:
        model.save_model(args.model_path)

    # Load the model
    # model.load_state_dict(torch.load(args.model_path))

    context = torch.zeros((1,1), dtype=torch.long, device=device)
    with open(args.output_path, 'a') as f:
        f.write(decode(model.generate(context, max_new_tokens=output_length)[0].tolist()) + '\n')

if __name__ == "__main__":
    main()