import csv
import ast
from transformers import T5Tokenizer, T5Model
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.long)
        }

# Load data from CSV file
train_x = []
train_y = []

with open('final_train_data1.csv', 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file)
    dict_for_target = {
        "NOT ENOUGH INFO": 0,
        "SUPPORTS": 1,
        "REFUTES": 2
    }
    for row in reader:
        claim, evidence, target = row
        if evidence == "[]" or evidence == []:
            list_of_evidence = []
        else:
            list_of_evidence = ast.literal_eval(evidence)  # Convert evidence string to list
        input_data = "[CLS] " + claim
        for ev in list_of_evidence:
            input_data += " [SEP] " + ev
        train_x.append(input_data)
        train_y.append(dict_for_target[target])  # Assuming target is numerical

# Instantiate the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Define custom T5 classification model
class CustomT5ForClassification(nn.Module):
    def __init__(self):
        super(CustomT5ForClassification, self).__init__()
        self.t5 = T5Model.from_pretrained("t5-small")
        self.classification_layer = nn.Linear(self.t5.config.d_model, 3)  # Assuming 3 output classes

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=torch.zeros(input_ids.shape).long().to(input_ids.device))
        last_hidden_state = outputs.last_hidden_state
        logits = self.classification_layer(last_hidden_state[:, 0])  # Take the representation of [CLS] token
        return logits

# Define hyperparameters
BATCH_SIZE = 32
MAX_LEN = 512  # Adjust as needed
num_epochs = 4  # Adjust as needed

# Create dataset and data loader
train_dataset = CustomDataset(train_x, train_y, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate the custom model
model = CustomT5ForClassification()

# Move model to appropriate device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.to(device)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the fine-tuned model if needed
torch.save(model.module.state_dict(), "t5_classification_model_new.pth")
