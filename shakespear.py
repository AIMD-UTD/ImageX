import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from torchtyping import TensorType
import pathlib


class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length:int, int_to_char: dict) -> str:
        generator = torch.Generator()
        generator.manual_seed(0)
        res = []
        # context is B x T
        # len(context) = B, len(context.T) = T
        # [5].item() -> 5
        for i in range(new_chars):
            if context.shape[1] > context_length:
                context = context[:, -context_length:]
            logits = model(context)
            last_time_step = logits[:, -1, :]
            probs = nn.functional.softmax(last_time_step, dim=-1)
            
            # The line where you call torch.multinomial(). Pass the generator as well
            next_char = torch.multinomial(probs, num_samples=1, generator=generator)
            
            context = torch.cat((context, next_char), dim = -1) # B, T -> B, T + !
            res.append(int_to_char[next_char.item()])
        return ''.join(res)


class Process:
        # You must start by generating batch_size different randome indices in the appropriate range
        # using a singe call to torch.randint()
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        torch.manual_seed(0)
        words = raw_dataset.split()
        indices = torch.randint(low=0, high = len(words) - context_length, size = (batch_size,))
        X = []
        Y = []
        for idx in indices:
            X.append(words[idx:idx+context_length])
            Y.append(words[idx+1:idx+1+context_length])
        return X,Y


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_block: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.pos_embeddings = nn.Embedding(context_length, model_dim)
        self.blocks = nn.Sequential()
        for i in range(num_block):
            self.blocks.append(self.TransformerBlock(model_dim, num_heads))
        self.final_ln = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)
    
    def forward(self, context: TensorType[int]) -> TensorType[float]:
        
        # Return logits, not probabilities
        token_embeds = self.token_embeddings(context) # B, T, D
        B, T, D = token_embeds.shape
        pos_embeds = self.pos_embeddings(torch.arange(T, device=context.device))
        total_embeddings = token_embeds + pos_embeds
        
        logits = self.vocab_projection(self.final_ln(self.blocks(total_embeddings)))
        return logits
        
    
    class TransformerBlock(nn.Module):
        
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.mhsa = self.MultiHeadedSelfAttention(model_dim, model_dim, num_heads)
            self.first_ln = nn.LayerNorm(model_dim)
            self.second_ln = nn.LayerNorm(model_dim)
            self.ff = self.VanillaNeuralNetwork(model_dim)
            
        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # Round answer to 4 decmal places
            
            first_part = embedded + self.mhsa(self.first_ln(embedded))
            res = first_part + self.ff(self.second_ln(first_part))
            return res
        
        class MultiHeadedSelfAttention(nn.Module):

            def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                # nn.ModuleList() will be useful. it works the same as a python list but is useful here since 
                # instance variables of any subclass of nn.Module must also be subclasses of nn.Module
                
                # use self.SingleHeatAttention to instantiate. You have to calculate head_size.
                self.heads = nn.ModuleList()
                for i in range(num_heads):
                    self.heads.append(self.SingleHeadAttention(embedding_dim, attention_dim // num_heads))
                
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                # Return answer to 4 decimal places
                outputs = [] # Each element in this list is B, T, Head_Size ->>>>> B, T, Attention_dim
                for head in self.heads:
                    outputs.append(head(embedded))
                cated = torch.cat(outputs, dim = 2)
                return cated

            class SingleHeadAttention(nn.Module):
                def __init__(self, embedding_dim: int, attention_dim: int):
                    super().__init__()
                    torch.manual_seed(0)
                    # output = w1*x1 + w2*x2 + w3*x3 + . . . ( no biasis because bias = false)
                    # b is a constant term, optionally learned through trining
                    
                    self.get_keys = nn.Linear(embedding_dim, attention_dim)
                    self.get_queries = nn.Linear(embedding_dim, attention_dim)
                    self.get_values = nn.Linear(embedding_dim, attention_dim)
                    
                def forward(self, embedded: TensorType[float]) -> TensorType[float]: # Return your answer to 4 decimal places
                    # Return your answer to 4 decimal places 
                    k = self.get_keys(embedded) # B, T, A
                    q = self.get_queries(embedded)
                    v = self.get_values(embedded)
                    
                    scores = q @ torch.transpose(k, 1, 2)
                    B, T, A = k.shape
                    scores = scores / (A ** 0.5)
                    
                    pre_mask = torch.tril(torch.ones(T, T, device=embedded.device))
                    mask = pre_mask == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=2) # B, T, T
                    transformed = scores @ v
                    return transformed

        class VanillaNeuralNetwork(nn.Module):
            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.linear1 = nn.Linear(model_dim, 4 * model_dim)
                self.linear2 = nn.Linear(4 * model_dim, model_dim)
                
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                
                x = F.relu(self.linear1(x))
                x = self.linear2(x)
                return x
            
            
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from torchtyping import TensorType
import pathlib


class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length:int, int_to_char: dict) -> str:
        generator = torch.Generator()
        generator.manual_seed(0)
        res = []
        # context is B x T
        # len(context) = B, len(context.T) = T
        # [5].item() -> 5
        for i in range(new_chars):
            if context.shape[1] > context_length:
                context = context[:, -context_length:]
            logits = model(context)
            last_time_step = logits[:, -1, :]
            probs = nn.functional.softmax(last_time_step, dim=-1)
            
            # The line where you call torch.multinomial(). Pass the generator as well
            next_char = torch.multinomial(probs, num_samples=1, generator=generator)
            
            context = torch.cat((context, next_char), dim = -1) # B, T -> B, T + !
            res.append(int_to_char[next_char.item()])
        return ''.join(res)


class Process:
        # You must start by generating batch_size different randome indices in the appropriate range
        # using a singe call to torch.randint()
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        torch.manual_seed(0)
        words = raw_dataset.split()
        indices = torch.randint(low=0, high = len(words) - context_length, size = (batch_size,))
        X = []
        Y = []
        for idx in indices:
            X.append(words[idx:idx+context_length])
            Y.append(words[idx+1:idx+1+context_length])
        return X,Y


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_block: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.pos_embeddings = nn.Embedding(context_length, model_dim)
        self.blocks = nn.Sequential()
        for i in range(num_block):
            self.blocks.append(self.TransformerBlock(model_dim, num_heads))
        self.final_ln = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)
    
    def forward(self, context: TensorType[int]) -> TensorType[float]:
        
        # Return logits, not probabilities
        token_embeds = self.token_embeddings(context) # B, T, D
        B, T, D = token_embeds.shape
        pos_embeds = self.pos_embeddings(torch.arange(T, device=context.device))
        total_embeddings = token_embeds + pos_embeds
        
        logits = self.vocab_projection(self.final_ln(self.blocks(total_embeddings)))
        return logits
        
    
    class TransformerBlock(nn.Module):
        
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.mhsa = self.MultiHeadedSelfAttention(model_dim, model_dim, num_heads)
            self.first_ln = nn.LayerNorm(model_dim)
            self.second_ln = nn.LayerNorm(model_dim)
            self.ff = self.VanillaNeuralNetwork(model_dim)
            
        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # Round answer to 4 decmal places
            
            first_part = embedded + self.mhsa(self.first_ln(embedded))
            res = first_part + self.ff(self.second_ln(first_part))
            return res
        
        class MultiHeadedSelfAttention(nn.Module):

            def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                # nn.ModuleList() will be useful. it works the same as a python list but is useful here since 
                # instance variables of any subclass of nn.Module must also be subclasses of nn.Module
                
                # use self.SingleHeatAttention to instantiate. You have to calculate head_size.
                self.heads = nn.ModuleList()
                for i in range(num_heads):
                    self.heads.append(self.SingleHeadAttention(embedding_dim, attention_dim // num_heads))
                
            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                # Return answer to 4 decimal places
                outputs = [] # Each element in this list is B, T, Head_Size ->>>>> B, T, Attention_dim
                for head in self.heads:
                    outputs.append(head(embedded))
                cated = torch.cat(outputs, dim = 2)
                return cated

            class SingleHeadAttention(nn.Module):
                def __init__(self, embedding_dim: int, attention_dim: int):
                    super().__init__()
                    torch.manual_seed(0)
                    # output = w1*x1 + w2*x2 + w3*x3 + . . . ( no biasis because bias = false)
                    # b is a constant term, optionally learned through trining
                    
                    self.get_keys = nn.Linear(embedding_dim, attention_dim)
                    self.get_queries = nn.Linear(embedding_dim, attention_dim)
                    self.get_values = nn.Linear(embedding_dim, attention_dim)
                    
                def forward(self, embedded: TensorType[float]) -> TensorType[float]: # Return your answer to 4 decimal places
                    # Return your answer to 4 decimal places 
                    k = self.get_keys(embedded) # B, T, A
                    q = self.get_queries(embedded)
                    v = self.get_values(embedded)
                    
                    scores = q @ torch.transpose(k, 1, 2)
                    B, T, A = k.shape
                    scores = scores / (A ** 0.5)
                    
                    pre_mask = torch.tril(torch.ones(T, T, device=embedded.device))
                    mask = pre_mask == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=2) # B, T, T
                    transformed = scores @ v
                    return transformed

        class VanillaNeuralNetwork(nn.Module):
            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.linear1 = nn.Linear(model_dim, 4 * model_dim)
                self.linear2 = nn.Linear(4 * model_dim, model_dim)
                
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                
                x = F.relu(self.linear1(x))
                x = self.linear2(x)
                return x
            
            


# Load your text file
with open("/Users/anishrajumapathy/transformer/shakespear.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build vocabulary
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}   # char -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int -> char

def encode_text(s: str):
    return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)

def decode_tokens(tokens: torch.Tensor):
    return "".join([itos[int(t)] for t in tokens])

# -----------------------------
# Minimal char-level training
# -----------------------------

# Define hyperparameters
input_text = "Shall I compare thee to a summer's day?"
vocab_size = len(stoi)
model_dim = 256
num_blocks = 4
num_heads = 8
encoded = encode_text(text).unsqueeze(0)  # use full text, not just a single line
context_length = 128  # keep as you had
# Create model
model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)



device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
encoded = encoded.to(device)

# Make simple sliding-window dataset from the encoded chars
def make_batches(data, context_length, batch_size):
    T = data.size(1)
    ix = torch.randint(0, T - context_length - 1, (batch_size,), device=device)
    X = torch.stack([data[:, i:i+context_length].squeeze(0) for i in ix])  # (B, T)
    Y = torch.stack([data[:, i+1:i+1+context_length].squeeze(0) for i in ix])  # (B, T)
    return X.long(), Y.long()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
batch_size = 64
train_steps = 1000  # bump to 10_000+ later

model.train()
for step in range(1, train_steps + 1):
    X, Y = make_batches(encoded, context_length, batch_size)
    logits = model(X)                          # (B, T, V)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),   # (B*T, V)
        Y.reshape(-1)                          # (B*T,)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}: loss {loss.item():.3f}")

# -----------------------------
# Sample after training
# -----------------------------
model.eval()
with torch.no_grad():
    def sample_with_logits(model, context, steps, context_length, itos, temperature=0.9, top_k=50):
        res = []
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(0)
        ctx = context.clone().to(device)
        for _ in range(steps):
            if ctx.shape[1] > context_length:
                ctx = ctx[:, -context_length:]
            logits = model(ctx)[:, -1, :] / temperature  # (1, V)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)  # (1,1)
            ctx = torch.cat([ctx, next_id], dim=1)
            res.append(itos[int(next_id.item())])
        return "".join(res)

    start_context = encoded[:, :context_length]
    generated = sample_with_logits(
        model, start_context, steps=400,
        context_length=context_length, itos=itos,
        temperature=0.9, top_k=50
    )

print("\nGenerated text after a bit of training:\n")
print(generated)


