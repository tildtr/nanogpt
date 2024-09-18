# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

