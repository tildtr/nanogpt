# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 
n_embd = 384 # embedding dimension
n_head = 6 # number of heads in the self-attention layer -> every head is 64-dimensional (384/6)
n_layer = 6 # number of transformer blocks
dropout = 0.2
# ----------------------------------------------

torch.manual_seed(1337)

# read input text file
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# get all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers - dictionaries
stoi = { ch:i for i,ch in enumerate(chars) } # stoi: string to integer
itos = { i:ch for i,ch in enumerate(chars) } # itos: integer to string

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Split into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long) # Tensor is a generalized form of a matrix that can have any number of dimensions
n = int(0.9 * len(data)) # first 90% for training
train_data, val_data = data[:n], data[n:]

# data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random starting indices for the sequences
    x = torch.stack([data[i:i+block_size] for i in ix]) # input sequences, stacked as rows in a tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # target sequences, shifted one position to the right, stacked as rows in a tensor
    x, y = x.to(device), y.to(device) 
    return x, y

# context manager to be more memory efficient as we do not intend to backpropagate through this code
@torch.no_grad()
# get the average loss over a few batches to be less noisy 
def estimate_loss():
    out = {}
    # set model to evaluation phase
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # set model back to train phase
    model.train()
    return out

# self attention block
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        # the query, key, and value matrices are all linear transformations of the input
        self.key = nn.Linear(n_embd, head_size, bias=False) # key vector - basic matrix multiply
        self.query = nn.Linear(n_embd, head_size, bias=False) # query vector
        self.value = nn.Linear(n_embd, head_size, bias=False) # value vector to align the dimensions to head_size 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix of block size

        self.dropout = nn.Dropout(dropout) # dropout layer to reduce overfitting

    def forward(self, x):
        B, T, C = x.shape # block size, time steps, head size
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T) - scaling by C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask out the upper triangular part of the matrix
        wei = F.softmax(wei, dim=-1) # (B, T, T) - softmax over the last dimension
        wei = self.dropout(wei) # apply dropout
        
        # weighted aggregation of values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out
    
class MultiHeadAttention(nn.Module):
    """ Implement multiple heads of attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # simply create multiple heads as defined
        self.proj = nn.Linear(n_embd, n_embd) # project the concatenated heads to the original embedding dimension
        self.dropout = nn.Dropout(dropout) # dropout layer to reduce overfitting   

    def forward(self, x): 
        # apply each head to the input and concatenate the results over the channel dimension 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # project the concatenated heads to the original embedding dimension
        return out
    
class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity"""
    # this is on the per token level
    # tokens gather data and than "start thinking" on that data

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer going back into the residual pathway, multiply by 4 to make the "side path" more experessive
            nn.Dropout(dropout), # dropout layer to reduce overfitting
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # self-attention block
        self.ffwd = FeedForward(n_embd) # feed-forward block
        self.ln1 = nn.LayerNorm(n_embd) # layer normalization
        self.ln2 = nn.LayerNorm(n_embd) # layer normalization

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # add "residual connection" to deal with the deepth of the NN, layernorm before forwarding
        x = x + self.ffwd(self.ln2(x)) # add "residual connection" to deal with the deepth of the NN
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        # Essentially, this layer is a lookup table that outputs a vector (of size vocab_size) for each token in the input. 
        # In this case, it's used to predict the next token (bigram prediction).
        # n_embd is number of embedding dimensions
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # get own embedding vector for position
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # 4 blocks of self-attention and feed-forward    
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both [batch_size, block_size] tensors of integers
        # The output is a tensor of shape [batch_size, block_size, vocab_size], where each value in the vocab_size dimension represents the score (logit) for a possible next token.
        # This means for each token in the input, the model produces a vector of vocab_size logits that represent the predicted probabilities of the next token in the sequence (the bigram prediction).
        
        # now this creates token embeddings instead of 
        tok_emb = self.token_embedding_table(idx) # [batch_size, block_size, vocab_size] or (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) - ints from 0 to T-1
        x = tok_emb + pos_emb # (B, T, C) -> now holds not only the token identities but also the positions at which they occur - currently useless as we have bigram model (will be necessary for self-attention block)
        #x = self.sa_heads(x) # (B, T, C) - self-attention block -> apply one self-attention head
        #x = self.ffwd(x) # (B, T, C) - feed-forward block
        x = self.blocks(x) # apply the blocks (B, T, C)
        x = self.ln_f(x) # apply final layer norm (B, T, C)
        logits = self.lm_head(x) # [batch_size, block_size, vocab_size] or (B, T, vocab_size)

        if targets is None:
            loss = None

        else:   
            # Inuitively, we want to predict the next token in the sequence, given the current token.
            # We can do this by comparing the predicted logits to the actual next token in the sequence.
            # This is done using the cross-entropy loss, which measures the difference between the predicted probabilities and the actual target (the next token in the sequence).
            
            # Cross entropy wants a (B, C, T) input, so we need to permute the logits tensor
            # It's mainly important that C is in second position so we just make 2-dimensional tensor like below
            # Basically stretching the array
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # This generate function does not make much sense for the Bigram model because Bigram only uses one character to predict the next
    # But we would like to keep it fixed to use with other more complex models
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            # need to crop the context if it's longer than block_size (idx)
            idx_cond = idx[:, -block_size:] # (B, min(T, block_size))

            # get the predictions for the next token
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B, C) as we remove the time dimension

            # apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the probability distribution to get the next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

model = BigramLanguageModel() # no need for vocab_size passing as it is already defined globally
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # index into the first batch and convert to list
