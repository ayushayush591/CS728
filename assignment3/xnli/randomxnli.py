import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google-t5/t5-large"
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

# Load datasets
dev_datasets = {}
test_datasets = {}
langs = ["en", "fr", "ru"]

for lang in langs:
    dataset = load_dataset("xnli", lang)
    dataset = dataset.map(lambda x: {"premise": x["premise"].strip()})
    dataset = dataset.map(lambda x: {"hypothesis": x["hypothesis"].strip()})
    dev_datasets[lang] = dataset["validation"]
    test_datasets[lang] = dataset["test"]

choices = {"en":["true", "unknown", "false"], "fr":["vrai", "inconnu", "faux"],"ru":["истина", "неизвестно", "ложь"]}

def format_example(premise, hypothesis, source_lang,label=None):
    prompt = "Premise: {premise}\nHypothesis: {hypothesis}".format(premise=premise, hypothesis=hypothesis)
    prompt += "\nAnswer:"
    if label is not None:
        label=choices[source_lang][int(label)]
        prompt += " {label}\n\n".format(label=label)
    return prompt

def gen_prompt(source_lang,dev_data, k=5):
    prompt = f"Here are few example of premise and hypothesis, we need to state that whether the hypothesis is is more likely to be true, false, or unknown based on the given premise.\n\n"
    random_numbers = random.sample(range(len(dev_data)), k)
    for number in random_numbers:
        label = dev_data[number]["label"]
        prompt += format_example(premise=dev_data[number]["premise"], hypothesis=dev_data[number]["hypothesis"], source_lang=source_lang,label=label)
    return prompt

# Calculate precision
def calculate_precision(pred_labels, true_labels):
    return precision_score(true_labels, pred_labels, average='macro')

precisions = []
precString=[]
for source_lang in langs:
    for target_lang in langs:
        if source_lang != target_lang:
            # Collect predictions and true labels
            pred_labels = []
            true_labels = []
            for i in range(200):
                few_shot_prompt = gen_prompt(source_lang,dev_data=dev_datasets[source_lang], k=2)
                query_input = test_datasets[target_lang][i]
                test_prompt = format_example(query_input['premise'], query_input['hypothesis'],source_lang)
                # prompt = f"Answer whether the hypothesis is more likely to be {choices[target_lang][0]}, {choices[target_lang][1]}, or {choices[target_lang][2]} based on the given premise.\n\n"
                prompt = f"Answer whether the hypothesis is more likely to be true, false, or unknown based on the given premise.\n\n"
                
                final_prompt = few_shot_prompt + prompt + test_prompt
                # print(final_prompt)
                prompt_inputs = tokenizer(final_prompt, return_tensors="pt").input_ids
                outputs = model.generate(prompt_inputs)
                
                pred_label = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
                pred_labels.append(pred_label)
                true_labels.append(choices['en'][int(query_input['label'])])
                # print(pred_label)
                # print(choices['en'][int(query_input['label'])])
                # exit()
                
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
