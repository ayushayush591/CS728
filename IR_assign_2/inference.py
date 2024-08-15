
import torch
from transformers import T5Tokenizer, T5Model
import torch.nn as nn
import json
import jsonlines
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from multiprocessing import Pool
# Load trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5Model.from_pretrained("t5-small")

def extract_named_entities(named_entities):
    entities = []
    for subtree in named_entities:
        if hasattr(subtree, 'label') and subtree.label():
            if subtree.label() in ['ORGANIZATION', 'PERSON', 'GPE', 'LOCATION']:
                entities.append(' '.join([child[0] for child in subtree]))
            else:
                entities.extend(extract_named_entities(subtree))
    return entities

def ner(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    named_entities = ne_chunk(tagged_words)
    return named_entities

# Define custom T5 classification model
class CustomT5ForClassification(nn.Module):
    def __init__(self):
        super(CustomT5ForClassification, self).__init__()
        self.t5 = model
        self.classification_layer = nn.Linear(self.t5.config.d_model, 3)  # Assuming 3 output classes

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=torch.zeros(input_ids.shape).long().to(input_ids.device))
        last_hidden_state = outputs.last_hidden_state
        logits = self.classification_layer(last_hidden_state[:, 0])  # Take the representation of [CLS] token
        return logits

# Load the trained model weights
model_path = "t5_classification_model_new.pth"
custom_model = CustomT5ForClassification()
custom_model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
custom_model.eval()

# Function for performing inference on a batch of texts
def classify_texts(input_texts):
    # Tokenize input texts
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate output
    with torch.no_grad():
        logits = custom_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Get predicted classes
    predicted_classes = torch.argmax(logits, dim=1).tolist()

    # Map numerical class index to label
    labels = {
        0: "NOT ENOUGH INFO",
        1: "SUPPORTS",
        2: "REFUTES"
    }
    predicted_labels = [labels[class_idx] for class_idx in predicted_classes]

    return predicted_labels

# input_text = "[CLS] Hotell is owned by Lisa Langseth. "
# print(classify_texts(input_text)[0])

result={}
i=0
with open('test_claim_evidence.json', 'r') as file:
    for line in file:
        json_data = json.loads(line)
        # Iterate through each dictionary in json_data
        for data in json_data:
            claim = data['claim']
            nerlist=ner(claim)
            entity_list = extract_named_entities(nerlist)
            flag=0
            # print(entity_list)
            for inst in entity_list:
                # print(i)
                if inst not in data['evidence']:
                    flag=1
            if flag==0:
                result[claim] =['NOT ENOUGH INFO', []]
                continue
            input_evidence = '[CLS] ' + claim + ' [SEP] ' + data['evidence']
            predicted_evidence = data['predicted_evidence']
            model_ouput=classify_texts(input_evidence)[0]
            result[claim] = [model_ouput, predicted_evidence]
            i+=1
            if i%100==0:
                print("iterator",i)
                # break
            # break

output_ranks=[]
i=0
with open('test.jsonl', 'r') as file:
    for line in file:
        json_data = json.loads(line)
        ids = json_data.get("id", "")
        claim = json_data.get("claim", "")
        output=result[claim]
        outputJson={}
        outputJson["id"]=ids
        outputJson["predicted_label"]=output[0]
        if output[0]=='NOT ENOUGH INFO':
            outputJson["predicted_evidence"]=[None]
        else:
            outputJson["predicted_evidence"]=[output[1]]
        output_ranks.append(outputJson)
        i+=1
        if i%100==0:
            print("iterator",i)
            # break
        # break
print(len(output_ranks))

output_file_path = 'predictions.jsonl'

# Write output_ranks to the JSONL file
with jsonlines.open(output_file_path, mode='w') as writer:
    for output in output_ranks:
        writer.write(output)