import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
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
            dim_feedforward=hidden_dim
        )

        # Create a Transformer encoder with multiple layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        embedded_input = self.embedding(input_ids)
        transformer_output = self.transformer_encoder(embedded_input)
        output_logits = self.fc(transformer_output[5,:,:])  # Assuming you want the last layer's output
        return output_logits

class MaskedGenerationDataset(Dataset):
    def __init__(self, train_data):
        self.data = train_data
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        triplet = self.data[idx]
        subject_id, relation_id, object_id = triplet[0], triplet[1], triplet[2]

        input_ids_masked = torch.tensor([0, sub[subject_id], 1, sub[relation_id], 1, 3, 2], dtype=torch.long)
        # tgt = torch.tensor([0, sub[subject_id], 1, sub[relation_id], 1, sub[object_id], 2], dtype=torch.long)
        tgt=torch.tensor([sub[object_id]])
        return input_ids_masked, tgt

# Create an instance of the custom dataset
dataset_train = MaskedGenerationDataset(train_data)
dataset_val = MaskedGenerationDataset(dev_data)
dataset_test = MaskedGenerationDataset(test_data)

# Create a DataLoader for masked generation approach
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
device="cpu"
# Initialize model, loss function, and optimizer
model = TransformerEncoderModel(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()  # -100 is the ignore_index for masked token
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for input_ids, target_ids in train_dataloader:
        output_logits = model(input_ids.view(-1, input_ids.shape[0]).to(device))
        loss = criterion(output_logits.to(device), target_ids.squeeze(1).to(device))  # Calculate loss directly
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}')
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_input_ids, val_target_ids in val_dataloader:
            optimizer.zero_grad()
            output_logits = model(val_input_ids.view(-1, val_input_ids.shape[0]).to(device))
            loss = criterion(output_logits.to(device), val_target_ids.squeeze(1).to(device))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss / len(val_dataloader)}')
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     early_stop_counter = 0
    # else:
    #     early_stop_counter += 1

    # if early_stop_counter >= early_stop_count:
    #     print(f'Early stopping after {epoch + 1} epochs as validation loss did not improve.')
    #     break
# Save the trained model
torch.save(model.state_dict(), f'transformer_model_5{dataset}.pth')