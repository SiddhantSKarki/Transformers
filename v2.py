import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64
block_size = 64
max_iterations = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
dropout = 0.2
n_layer = 6
n_head = 6

# Random State for reproducability 
torch.manual_seed(1337)


# Reading the input data
with open("./input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers (Token encoding/Decoding)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: "".join([itos[i] for i in x])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# Utility Functions for 

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - batch_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


###################################### Model Architerture Components Starts Here ##################################################
###################################################################################################################################

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute Attention scores ("Affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # Randomly preventing some nodes from communicating
        # Takes your NN, and randomly shuts off some subsets in every forward and backward pass
        # THis ends up kind of training ensamble of sub networks
        # WHile inference this regularization not going to work.
        wei = self.dropout(wei)

        # Perform the weighted aggregration of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # A Linear transformation of the outcome of the heads layer (Multiheads)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out




# Computation being on a per-node level
class FeedForward(nn.Module):
    """
        After the MultiHead Attention:
            - Tokens looked at each other but didn't had enough time to think what they found
        This is a single layer FeedForward Layer
    
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # We're growing the layer
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)




# Now what we'd like to do is intersperse communication
# MANY MANY times
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        # We fork off and do some communication and come bacl
        # Doing so adds gradients equally up to the original input during back propogation
        # Addtiona distributes gradiesnt equally to both the branches
        # Gradients hop to every additional node
        # There is a gradient superhighway 
        # In the beginning they contribute very less
        # But during optmization, then the block over time kick in
        # Original paper does this a bit differently 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Language modelling head
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb
        x = self.blocks(x) 
        x = self.ln_f(x)
        # x = self.sa_heads(x)
        # Here x not only holds token identities, but also the positions where these occur
        # in case of bigram model, the position embedding is not that useful
        # x = self.ffwd(x)
        # Self attention is the communication 
        # Once they've gathered all the info, now they think on that data indivudually (per token level).
        logits = self.lm_head(x) # (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cropped = idx[:, -block_size:]

            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
###################################### Model Architerture Components End Here ##################################################
###################################################################################################################################

if __name__ == "__main__":
    model = BigramLanguageModel()
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iterations):

        if iter % eval_interval == 0 or iter == max_iterations - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses["train"]:0.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1,1), dtype=torch.long, device=device)
    out = decode(m.generate(context, max_tokens=10000)[0].tolist())
    with open("output.txt", 'w') as file:
        file.write(out)
