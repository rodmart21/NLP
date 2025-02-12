import re
import unicodedata
import random
from sqlalchemy import create_engine
from langdetect import detect, LangDetectException, DetectorFactory
import openai
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import unicodedata
import json
import os
import json

from src.classes import *
from src.utils import *
from src.constants import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "bayer_whole_cat.csv")
key_path = os.path.join(BASE_DIR, "keys", "config.json")
dict_files = [os.path.join(BASE_DIR, "dictionaries", "dict_de.json"),os.path.join(BASE_DIR, "dictionaries", "dict_en.json"),os.path.join(BASE_DIR, "dictionaries", "dict_es.json")]

with open(key_path, "r") as config_file:
    config = json.load(config_file)

openai.api_key =config["api_key"]

# Example usage
merged_dict = merge_dictionaries(load_dictionaries(dict_files))

# Database connection setup
# db_username = config["db_username"]
# db_host = config["db_host"]
# db_pass = config["db_pass"]
# db_name = config["db_name"]

# noun_extractor = DatabaseConnector(db_username, db_host, db_pass, db_name)
# view_name = config["view_name"]
# bayer_catalog = noun_extractor.import_view(view_name)

bayer_catalog=pd.read_csv(csv_path)
bayer_catalog_test = bayer_catalog[20:25]
print(bayer_catalog_test[['short_description', 'noun']])

fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:sparrow::As7oH6WH"
system_prompt = "You are a helpful assistant for product classification."
classifier = ProductClassifier(fine_tuned_model=fine_tuned_model, system_prompt=system_prompt, allowed_values=allowed_category_values)

cleaned_df = classifier.clean_data(bayer_catalog_test, 'noun', clean_text, match_terms_exact, merged_dict)
df_matched = cleaned_df.loc[cleaned_df['matched_category_clean'] != 'Unmatched']
df_unmatched = cleaned_df[cleaned_df['matched_category_clean'] == 'Unmatched']

predictions, df_model = classifier.classify_unmatched(df_unmatched=df_unmatched, predict_categories=predict_categories)
print(df_model)

# df_final=concat...
# output_table=...
# noun_extractor.upload_to_database(df_final, output_table)