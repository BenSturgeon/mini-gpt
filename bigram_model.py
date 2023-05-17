import torch
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import urllib.request

# define variables


batch_size = 32
block_size = 8
n_embd = 32
lr = 1e-3
training_iters = 3000
eval_interval = 300
eval_iters = 200
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
def estimate_loss():
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
        

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(x_input) # we're not using loss, as we're generating

            next_token = logits[:, -1,:]

            probabilities = F.softmax(next_token, dim=-1)

            top_answer = torch.multinomial(probabilities, num_samples=1)

            x_input = torch.cat((x_input, top_answer), dim=1) # B, T+1. Appending to 1st dimension which is the time dimension

        return x_input
        

model = BigramLanguageModel(vocab_size)
model =model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(training_iters):

    if iter % eval_interval == 0:
        averaged_losses = estimate_loss()
        print(f"steps: {iter}  train loss:{averaged_losses['train']:.4f}  test loss:{averaged_losses['val']:.4f}")

    xb, yb = get_batch(batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))