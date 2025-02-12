import openai
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import unicodedata
import re
import os
import random
import json
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# ================================ TEXT CLEANING FUNCTIONS ================================

def clean_text(text):
    """
    General text cleaning function: removes accents, punctuation, and normalizes whitespace.
    """
    cleaned_text = str(text).strip().lower()
    cleaned_text = unicodedata.normalize('NFD', cleaned_text)
    cleaned_text = re.sub(r'[\u0300-\u036f]', '', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text


def clean_header(text):
    """
    Cleans header text for better comparison: removes double spaces and capitalizes words.
    """
    return " ".join(text.split()).title()


def count_words(text):
    """
    Counts words in a string.
    """
    return len(text.split()) if isinstance(text, str) else 0


# ================================ LANGUAGE DETECTION ================================

def detect_language(text):
    """
    Detects the language of a given text.
    """
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


# ================================ MODEL PREDICTION FUNCTIONS ================================

def predict_categories(fine_tuned_model, input_texts, system_prompt):
    """
    Uses OpenAI's fine-tuned model to classify text inputs.
    """
    predictions = []
    for text in input_texts:
        try:
            response = openai.ChatCompletion.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            predictions.append(response['choices'][0]['message']['content'].strip())
        except Exception as e:
            predictions.append(f"Error: {str(e)}")
    return predictions


# ================================ METRICS CALCULATION ================================


def compute_metrics(true_labels, predicted_labels):
    """
    Computes accuracy, classification report, and confusion matrix.
    Formats the classification report for better readability.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, digits=2)
    matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Formatting the output
    formatted_report = f"Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}"
    
    return {"accuracy": accuracy, "classification_report": formatted_report, "confusion_matrix": matrix}


# ================================ MERGING DICTIONARIES ================================


def load_dictionaries(dict_files, base_dir=None):
    """
    Load and return a list of JSON dictionaries from a directory.
    """
    if base_dir is None:
        base_dir = os.getcwd()  # Use current working directory instead of __file__
    dict_dir = os.path.join(base_dir, "dictionaries")
    
    return [json.load(open(os.path.join(dict_dir, file), "r", encoding="utf-8"))
            for file in dict_files if os.path.exists(os.path.join(dict_dir, file))]

def merge_dictionaries(dict_list):
    """
    Merge multiple dictionaries into one.
    """
    return {k: v for d in dict_list for k, v in d.items()}


# ================================ CATEGORY MATCHING FUNCTIONS ================================

def match_terms_exact(noun, pattern_dict, category_dict, cat_unmatched):
    """
    Matches terms exactly with predefined dictionary keys.
    """
    return (category_dict.get(noun, cat_unmatched), noun if noun in pattern_dict else None)


def match_terms_not_exact(noun, pattern_dict, category_dict, cat_unmatched):
    """
    Matches terms using regex patterns.
    """
    for term, pattern in pattern_dict.items():
        if pattern.search(str(noun)):
            return category_dict[term], term
    return cat_unmatched, None


def match_terms_vectorized(noun, pattern_dict, category_dict):
    """
    Vectorized matching function for efficiency.
    """
    return next(((category_dict[term], term) for term, pattern in pattern_dict.items() if pattern.search(str(noun))), (None, None))


# ================================ DATA PROCESSING AND METRICS ================================

def process_and_compute_metrics(df_matched, df_unmatched, qa_category=None):
    """
    Combines matched and unmatched data and computes final metrics.
    """
    df_combined = pd.concat([df_matched, df_unmatched], ignore_index=True)
    df_combined['final_category'] = df_combined['matched_category_clean'].replace("Unmatched", None).combine_first(df_combined['predictions_model_clean'])
    
    if qa_category and qa_category in df_combined.columns:
        df_combined["qa_category_clean"] = df_combined[qa_category].apply(clean_header)
        df_combined = df_combined[df_combined['qa_category_clean'] != 'Unclassified']
        true_labels = df_combined['qa_category_clean'].tolist()
        predicted_labels = df_combined['final_category'].tolist()
        metrics = compute_metrics(true_labels, predicted_labels)
    else:
        metrics = None
    
    return df_combined, metrics


# ================================ DATA TRANSFORMATION FUNCTIONS ================================

def transform_to_messages_format(keys, dictionary, prompt_type=1):
    """
    Transforms data into OpenAI message format.
    """
    system_prompts = {
        1: "You are a helpful assistant for product classification.",
        2: "You are a helpful assistant for product classification. You classify products. You can only use the same categories that are in the training data. You will have short descriptions as inputs so you can have more context on every part you have to categorize."
    }
    return [{
        "messages": [
            {"role": "system", "content": system_prompts[prompt_type]},
            {"role": "user", "content": key},
            {"role": "assistant", "content": dictionary[key]}
        ]
    } for key in keys]


# ================================ CATEGORY SAMPLING & MODIFICATION ================================

def sample_categories(dictionary, samples):
    """
    Samples a specified number of items from each category.
    """
    category_to_items = {}
    for key, value in dictionary.items():
        category_to_items.setdefault(value, []).append(key)
    
    sampled_data = []
    for category, num_samples in samples.items():
        if category in category_to_items:
            sampled_data.extend(random.sample(category_to_items[category], min(num_samples, len(category_to_items[category]))))
    
    return sampled_data


def modify_category(data, add_category, remove_category, words_to_remove, words_to_add):
    """
    Modifies categories by removing and adding specific words.
    """
    if remove_category in data:
        data[remove_category] = [word for word in data[remove_category] if word not in words_to_remove]
    
    if add_category in data:
        data[add_category].extend(words_to_add)
    
    return data
