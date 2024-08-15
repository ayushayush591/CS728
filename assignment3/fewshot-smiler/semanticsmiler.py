import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity

model_name = "google-t5/t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load datasets
langs = ["en", "fr", "ru"]
dev_datasets = {}
test_datasets = {}

for lang in langs:
    dev_dataset = pd.read_csv(lang+'_corpora_train.tsv', sep='\t').drop(columns=['id', 'lang'])
    dev_dataset = dev_dataset.head(1000)
    test_dataset = pd.read_csv(lang+'_corpora_test.tsv', sep='\t').drop(columns=['id', 'lang'])
    dev_datasets[lang] = dev_dataset.values.tolist()
    test_datasets[lang] = test_dataset.values.tolist()

choice_array = [
    "has-genre", "has-type", "has-parent", "invented-by", "is-member-of", "headquarters", 
    "has-occupation", "has-author", "invented-when", "from-country", "birth-place", 
    "movie-has-director", "org-has-founder", "has-population", "org-has-member", "has-edu", 
    "has-nationality", "is-where", "starring", "org-leader", "has-spouse", "has-sibling", 
    "won-award", "loc-leader", "has-child", "event-year", "has-weight", "has-height", 
    "has-length", "has-highest-mountain", "first-product", "has-tourist-attraction", 
    "has-lifespan", "no_relation", "eats", "post-code"
]

def compute_embeddings(data):
    embeddings = []
    for example in data:
        input_text = example[-1]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, decoder_input_ids=inputs.input_ids)
        
        last_hidden_state = outputs[0]  # First element of the outputs tuple contains the hidden states
        embeddings.append(last_hidden_state.mean(dim=1).squeeze())  # Taking mean of the last hidden states
    
    return torch.stack(embeddings)

def format_example(text, entity1, entity2, label=None):
    prompt = f"Text: {text}\n Entity1: {entity1} \n Entity2: {entity2}\nRelation between entities is :\n"
    if label is not None:
        prompt += f" {label}\n"
    return prompt

def gen_prompt(source_lang,test_instance, dev_data, k=5):
    test_embedding = compute_embeddings([test_instance[-1]])[0]
    
    # Compute cosine similarity between the test instance and all development examples
    similarities = cosine_similarity(test_embedding.reshape(1, -1), dev_embeddings)[0]
    
    # Get indices of top k most similar examples
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    prompt = f"Here are a few examples of text, entity 1, entity 2, and the relation between entity 1 and entity 2.\n\n"
    for number in top_k_indices:
        label = dev_data[number][-2]
        prompt += format_example(text=dev_data[number][-1], entity1=dev_data[number][0], entity2=dev_data[number][1], label=label)
    prompt += f"\nFor each example, specify the relationship between the given entities from the list: {', '.join(choice_array)}\n\n"
    return prompt

# Calculate precision
def calculate_precision(pred_labels, true_labels):
    return precision_score(true_labels, pred_labels, average='macro')

precisions = []
precString=[]
for source_lang in langs:
    dev_embeddings = compute_embeddings(dev_datasets[source_lang])
    for target_lang in langs:
        if source_lang != target_lang:
            pred_labels = []
            true_labels = []
            for i in range(100):
                query_input = test_datasets[target_lang][i]
                few_shot_prompt = gen_prompt(source_lang, test_instance=query_input,dev_data=dev_datasets[source_lang], k=2)
                
                test_prompt = format_example(query_input[-1], query_input[0], query_input[1])
                prompt = f"Answer the relationship between entities for the given text:\n"
                final_prompt = f"{few_shot_prompt}{prompt}{test_prompt}".replace("<e1>", '').replace("<e2>", '').replace("</e1>", '').replace("</e2>", '')
                # print(final_prompt)
                prompt_inputs = tokenizer(final_prompt, return_tensors="pt").input_ids
                outputs = model.generate(prompt_inputs)
                pred_label = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
                pred_labels.append(pred_label)
                true_labels.append(query_input[-2])
                # print(pred_label,query_input[-2])
                # exit()
                
            precision = calculate_precision(pred_labels, true_labels)
            print(f"Precision for {source_lang}-{target_lang}: {precision}")
            precString.append(f"Precision for {source_lang}-{target_lang}: {precision}")
            precisions.append(precision)

# Calculate overall precision
overall_precision = np.mean(precisions)
for val in precString:
    print(val)
print(f"\nOverall precision across all language pairs: {overall_precision}")