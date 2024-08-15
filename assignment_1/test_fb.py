# %%
from datasets import load_dataset
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed()

data = open(f'./datasets/fb15k/train.txt', 'r').read().split('\n')
data_dev = open(f'./datasets/fb15k/valid.txt', 'r').read().split('\n')
data_test = open(f'./datasets/fb15k/test.txt', 'r').read().split('\n')
train_data = []
val_data = []
test_data = []
from collections import defaultdict
sub_dict = defaultdict(int)
target_dict = defaultdict(int)

for i in data[:-1]:
    train_data.append(i.split("\t"))
    sub_dict[i.split("\t")[0]] += 1
    sub_dict[i.split("\t")[1]] += 1
    sub_dict[i.split("\t")[2]] += 1
    target_dict[i.split("\t")[2]] += 1

for i in data_dev[:-1]:
    val_data.append(i.split("\t"))
    sub_dict[i.split("\t")[0]] += 1
    sub_dict[i.split("\t")[1]] += 1
    sub_dict[i.split("\t")[2]] += 1
    target_dict[i.split("\t")[2]] += 1
for i in data_test[:-1]:
    test_data.append(i.split("\t"))
    sub_dict[i.split("\t")[0]] += 1
    sub_dict[i.split("\t")[1]] += 1
    sub_dict[i.split("\t")[2]] += 1        
    target_dict[i.split("\t")[2]] += 1

sub = {j : i + 4 for i, j in enumerate(sub_dict)}
target_dict = {j : i + 4 for i, j in enumerate(target_dict)}
sub['[CLS]'] = 0
sub['[SEP]'] = 1
sub['[END]'] = 2
sub['[MASK]'] = 3

vocab_size = len(sub)
target_size = len(target_dict)
d_model = 128
num_heads = 8
hidden_dim = 512
num_layers = 4
batch_size = 32
learning_rate = 0.001
num_epochs = 100
max_len = 7
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# %%
class Embedding(nn.Module):
   def __init__(self):
       super(Embedding, self).__init__()
       self.tok_embed = nn.Embedding(vocab_size, d_model)
       self.pos_embed = nn.Embedding(max_len, d_model)
       self.norm = nn.LayerNorm(d_model)

   def forward(self, x):
       seq_len = x.size(1)
       pos = torch.arange(seq_len, dtype=torch.long).to(device)
       pos = pos.unsqueeze(0).expand_as(x)
       embedding = self.tok_embed(x) + self.pos_embed(pos)
       return self.norm(embedding)

# %%
class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers,dropout: float = 0.5):
        super(TransformerEncoderModel, self).__init__()
        self.embedding = Embedding()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, mask_check=False)

        self.classifier = nn.Linear(embedding_dim, target_size)

    def forward(self, input_ids):
        src = self.embedding(input_ids)
        transformer_output = self.transformer_encoder(src)
        output_logits = self.classifier(transformer_output[:, -2, :])
        return output_logits

# %%
class MaskedGenerationDataset(Dataset):
    def __init__(self, train_data):
        self.data = train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triplet = self.data[idx]
        subject_id, relation_id, object_id = triplet[0], triplet[1], triplet[2]

        input_ids_masked = torch.tensor([0, sub[subject_id], 1, sub[relation_id], 1, 3, 2], dtype=torch.long)
        tgt = torch.tensor(target_dict[object_id])
        return input_ids_masked, tgt

# %%
# Create an instance of the custom dataset
dataset_train = MaskedGenerationDataset(train_data)
dataset_val = MaskedGenerationDataset(val_data)
dataset_test = MaskedGenerationDataset(test_data)

# Create a DataLoader for masked generation approach
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = TransformerEncoderModel(vocab_size, d_model, num_heads, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
model.load_state_dict(torch.load('transformer_model_fn.pth'))
model.to(device)

# %%
# Initialize variables
total_hits = 0
total_hits1 = 0
total_mrr = 0
total_map = 0

with torch.no_grad():
    for test_input_ids, test_target_ids in test_dataloader:
        output_logits = model(test_input_ids.to(device))
        _, indices = torch.topk(output_logits, 1, dim=1)
        
        # Calculate Hits@1
        hits1 = torch.sum(indices == test_target_ids.unsqueeze(1).to(device), dim=1)
        total_hits1 += torch.sum(hits1 > 0).item()

        _, indices = torch.topk(output_logits, 10, dim=1)

        # Calculate Hits@10
        hits = torch.sum(indices == test_target_ids.unsqueeze(1).to(device), dim=1)
        total_hits += torch.sum(hits > 0).item()

        # Calculate MRR
        reciprocal_ranks = torch.zeros_like(test_target_ids, dtype=torch.float)
        for i in range(len(test_target_ids)):
            rank = torch.where(indices[i] == test_target_ids[i])[0]
            if len(rank) > 0:
                reciprocal_ranks[i] = 1.0 / (rank[0].item() + 1)
        total_mrr += torch.sum(reciprocal_ranks).item()

        # Calculate MAP
        avg_precision = torch.zeros_like(test_target_ids, dtype=torch.float)
        for i in range(len(test_target_ids)):
            num_hits = 0
            precision_at_i = 0
            for j, index in enumerate(indices[i]):
                if index == test_target_ids[i]:
                    num_hits += 1
                    precision_at_i += num_hits / (j + 1)
            if num_hits > 0:
                avg_precision[i] = precision_at_i / num_hits
        total_map += torch.sum(avg_precision).item()

# Calculate final scores
total_samples = len(test_dataloader.dataset)
hits_at_10 = total_hits / total_samples
hits_at_1 = total_hits1 / total_samples
mrr = total_mrr / total_samples
map_score = total_map / total_samples

hits_at_1, hits_at_10, mrr, map_score



