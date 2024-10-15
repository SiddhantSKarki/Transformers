import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(42);

# Reading the text data
text = open("./data/11-0.txt", 'r', encoding='UTF-8').read()[664:]
chars = "". join(sorted(list(set(text))))
print(f"Vocab Size: {len(chars)}")

# Hyperparameters
batch_size = 16
block_size = 8
vocab_size = len(chars)
epochs = 5000
log_iter = 100
eval_ter = 200
learning_rate = 1e-4
n_embd = 64
num_heads = 4
n_blocks = 8
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"


# Utility Functions : Dataloader
# Creating a dataloader from scratch

# Encoding and Decoding Mapping
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

# Encoding and Decoding Lambdas
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: "".join([itos[token] for token in x])

data = torch.tensor(encode(text), dtype=torch.long)
split_size = 0.9
n = int(split_size * len(data))
train_set = data[:n]
val_set = data[n:]

def get_batches(split):
    batch_data = train_set if split == "train" else val_set
    idxs = torch.randint(len(data) - block_size - 1, (batch_size, ))
    x = torch.stack([data[i:block_size+i] for i in idxs])
    y = torch.stack([data[i+1:block_size+i+1] for i in idxs])
    x, y = x.to(device), y.to(device)
    return x, y
############################################################################################
####################<-----Transformer Architecture Code Starts Here-------->################

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Calculating the affinities
        wei = (q @ k.transpose(-2, -1)) * k.shape[-1]**0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

# Till MultiHeadAttention we're still extracting feature (how are tokens reacting to each other
# How much of "Attention" is one token emmitting to the other
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.multi_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        ''' 
            We fork off and do some communication and come back.
            Doing so adds gradients equally up to the original input during back propogation.
            Addtionally distributes gradient equally to both the branches.
            Gradients hop to every additional node.
            There is a gradient superhighway .
            In the beginning they contribute very less.
            But during optimization, then the block over time kick in
            Original paper does this a bit differently 
        '''
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

class GPT2Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # pluck out the learned token embedding vectors for each token
        token_embds = self.embedding_table(idx)
        # pluck out the learned position embedding vectors for each token
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = token_embds + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return loss, logits 
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cropped = idx[:, -block_size:]
            loss, logits = self(idx_cropped)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # concat along the time dimension
        return idx
    
####################<-----Transformer Architecture Code Ends Here-------->################    
############################################################################################
def train():
    model = GPT2Transformer()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_ter)
            for k in range(eval_ter):
                X, Y = get_batches(split)
                loss, logits = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(epochs):
        if iter % log_iter == 0 or iter == epochs - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses["train"]:0.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batches("train")
        loss, logits = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model



if __name__ == "__main__":
    trained_model = train()
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    output = trained_model.generate(context, max_tokens=10000)[0].tolist()

    open("outputs.txt", "w").write(decode(output))
