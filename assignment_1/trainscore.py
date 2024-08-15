import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="wn18",
    help="if specified, we will load the tokenizer from here.",
)
args = parser.parse_args()
dataset=args.dataset
data = open(f'./datasets/{dataset}/train.txt', 'r').read().split('\n')
data_dev = open(f'./datasets/{dataset}/valid.txt', 'r').read().split('\n')
data_test = open(f'./datasets/{dataset}/test.txt', 'r').read().split('\n')
train_data = []
dev_data = []
test_data = []
print("hello")
from collections import defaultdict
sub_dict = defaultdict(int)
for i in data[:-1]:
    train_data.append(i.split("\t"))
    sub_dict[i.split("\t")[0]] += 1
    sub_dict[i.split("\t")[1]] += 1
    sub_dict[i.split("\t")[2]] += 1
for i in data_dev[:-1]:
    dev_data.append(i.split("\t"))
    sub_dict[i.split("\t")[0]] += 1
    sub_dict[i.split("\t")[1]] += 1
    sub_dict[i.split("\t")[2]] += 1
for i in data_test[:-1]:
    test_data.append(i.split("\t"))
    sub_dict[i.split("\t")[0]] += 1
    sub_dict[i.split("\t")[1]] += 1
    sub_dict[i.split("\t")[2]] += 1        

sub = {j : i+4 for i, j in enumerate(sub_dict)}
sub['[CLS]'] = 0
sub['[SEP]'] = 1
sub['[END]'] = 2
sub['[MASK]'] = 3
print("hello 1")
vocab_size = len(sub)  # Example vocabulary size
embedding_dim = 256
num_heads = 4
hidden_dim = 512
num_layers = 4
batch_size = 32
learning_rate = 0.0001
num_epochs = 100

# Early stopping parameters
early_stop_count = 3  # Number of consecutive epochs with no improvement after which training will stop
best_val_loss = float('inf')
early_stop_counter = 0

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define a single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True  # Add this line
        )


        # Create a Transformer encoder with multiple layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)
    def forward(self, input_ids):
            embedded_input = self.embedding(input_ids)
            transformer_output = self.transformer_encoder(embedded_input)
            scores = self.fc(transformer_output[:, :])  # Fix here
            return scores
    
class ScoreBasedDataset(Dataset):
    def __init__(self, train_data):
        self.data = train_data
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        triplet = self.data[idx]
        subject_id, relation_id, object_id = triplet[0], triplet[1], triplet[2]

        positive_tgt = torch.tensor([sub[object_id]])

        negative_object_id = self.generate_negative_object_id(object_id)
        negative_tgt = torch.tensor([sub[negative_object_id]])

        return positive_tgt, negative_tgt

    def generate_negative_object_id(self, positive_object_id):
        # Randomly select an object ID different from the positive object ID
        candidate_ids = list(sub.keys())
        candidate_ids.remove(positive_object_id)
        return random.choice(candidate_ids)


print("hello 2")
# Create an instance of the custom dataset
dataset_train = ScoreBasedDataset(train_data)
dataset_val = ScoreBasedDataset(dev_data)
dataset_test = ScoreBasedDataset(test_data)

# Create a DataLoader for masked generation approach
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
device="cpu"
# Initialize model, loss function, and optimizer
model = TransformerEncoderModel(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers)
criterion = nn.CosineEmbeddingLoss()  # You can use nn.TripletMarginLoss for triplet loss
print("hello 3")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)
margin = 0.1
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    print("4")
    print(train_dataloader)
    for positive_tgt, negative_tgt in train_dataloader:
        
        positive_scores = model(positive_tgt.squeeze(1).to(device))  # Squeeze the additional dimension
        negative_scores = model(negative_tgt.squeeze(1).to(device))  # Squeeze the additional dimension

        target = torch.ones(positive_scores.shape[0]).to(device)
        loss = criterion(positive_scores, negative_scores, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}')

# Save the model after training
torch.save(model.state_dict(), f'transformer_model_score_based_{dataset}.pth')