import numpy as np
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score
model_name = "google-t5/t5-large"
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

# Load datasets
dev_datasets = {}
test_datasets = {}
langs = ["en", "fr", "ru", "zh"]

for lang in langs:
    dataset = load_dataset("xnli", lang)
    dataset = dataset.map(lambda x: {"premise": x["premise"].strip()})
    dataset = dataset.map(lambda x: {"hypothesis": x["hypothesis"].strip()})
    dev_datasets[lang] = dataset["validation"]
    test_datasets[lang] = dataset["test"]

choices = {"en":["true", "unknown", "false"], "fr":["vrai", "inconnu", "faux"],"ru":["истина", "неизвестно", "ложь"], "zh":["真", "未知", "假"]}

def format_example(source_lang,premise, hypothesis, label=None):
    prompt = "Premise: {premise}\nHypothesis: {hypothesis}".format(premise=premise, hypothesis=hypothesis)
    prompt += "\nAnswer:"
    if label is not None:
        label=choices[source_lang][int(label)]
        prompt += " {label}\n\n".format(label=label)
    return prompt

def compute_embeddings(data):
    embeddings = []
    for example in data:
        input_text = example["premise"] + " " + example["hypothesis"]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, decoder_input_ids=inputs.input_ids)
        
        last_hidden_state = outputs[0]  # First element of the outputs tuple contains the hidden states
        embeddings.append(last_hidden_state.mean(dim=1).squeeze())  # Taking mean of the last hidden states
    
    return torch.stack(embeddings)


def gen_prompt(source_lang,dev_embeddings, dev_data, test_instance, k=2):
    
    test_embedding = compute_embeddings([test_instance])[0]
    
    # Compute cosine similarity between the test instance and all development examples
    similarities = cosine_similarity(test_embedding.reshape(1, -1), dev_embeddings)[0]
    
    # Get indices of top k most similar examples
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    prompt = f"Here are a few examples of premise and hypothesis related to the test instance:\n\n"
    for index in top_k_indices:
        example = dev_data[int(index)]  # Convert numpy index to Python integer
        prompt += format_example(source_lang=source_lang,premise=example["premise"], hypothesis=example["hypothesis"], label=example.get("label"))
    
    return prompt


# Example usage
# source_lang = "en"
# target_lang = "fr"
# test_instance = test_datasets[target_lang][0]  # Choose a test instance
# dev_embeddings = compute_embeddings(dev_datasets[source_lang])
# few_shot_prompt = gen_prompt(source_lang=source_lang,dev_embeddings=dev_embeddings,dev_data=dev_datasets[source_lang], test_instance=test_instance, k=2)
# print(few_shot_prompt)
# print(test_instance)



def calculate_precision(pred_labels, true_labels):
    return precision_score(true_labels, pred_labels, average='macro')

precisions = []
precString=[]
for source_lang in langs:
    dev_embeddings = compute_embeddings(dev_datasets[source_lang])
    for target_lang in langs:
        if source_lang != target_lang:
            # Collect predictions and true labels
            pred_labels = []
            true_labels = []
            for i in range(200):
                query_input = test_datasets[target_lang][i]
                few_shot_prompt = gen_prompt(source_lang=source_lang,dev_embeddings=dev_embeddings,dev_data=dev_datasets[source_lang], test_instance=query_input, k=2)
                
                test_prompt = format_example(source_lang,query_input['premise'], query_input['hypothesis'])
                prompt = f"Answer whether the hypothesis is more likely to be true, false, or unknown based on the given premise.\n\n"
                final_prompt = few_shot_prompt + prompt + test_prompt
                # print(final_prompt)
                prompt_inputs = tokenizer(final_prompt, return_tensors="pt").input_ids
                outputs = model.generate(prompt_inputs)
                
                pred_label = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
                pred_labels.append(pred_label)
                true_labels.append(choices['en'][int(query_input['label'])])
                
            # Calculate precision for the current language pair
            precision = calculate_precision(pred_labels, true_labels)
            print(f"Precision for {source_lang}-{target_lang}: {precision}")
            precString.append(f"Precision for {source_lang}-{target_lang}: {precision}")
            precisions.append(precision)

# Calculate overall precision
overall_precision = np.mean(precisions)
for val in precString:
    print(val)
print(f"\nOverall precision across all language pairs: {overall_precision}")
