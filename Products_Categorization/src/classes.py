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

class DatabaseConnector:
    def __init__(self, db_username, db_host, db_pass, db_name):
        """Initialize database connection."""
        self.db_username = db_username
        self.db_host = db_host
        self.db_pass = db_pass
        self.db_name = db_name
        self.engine = self.create_connection()

    def create_connection(self):
        """Establish a database connection using SQLAlchemy."""
        try:
            connection_string = f'postgresql://{self.db_username}:{self.db_pass}@{self.db_host}:5432/{self.db_name}'
            engine = create_engine(connection_string)
            print("Database connection successful.")
            return engine
        except Exception as e:
            print(f"Error creating database connection: {e}")
            return None

    def import_view(self, view_name):
        """Import data from a database view."""
        try:
            query = f"SELECT * FROM {view_name}"
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            print(f"Data imported successfully from view: {view_name}")
            return df
        except Exception as e:
            print(f"Error importing data from view: {e}")
            return None

    def upload_to_database(self, df, table_name, schema='processing'):
        """Upload a DataFrame to the database."""
        try:
            df.to_sql(table_name, self.engine, if_exists='append', index=False, schema=schema)
            print(f"Data uploaded successfully to {schema}.{table_name}")
        except Exception as e:
            print(f"Error uploading data to database: {e}")


class ProductClassifier:
    def __init__(self, fine_tuned_model, system_prompt, allowed_values):
        """Initialize ProductClassifier with model details and allowed categories."""
        self.fine_tuned_model = fine_tuned_model
        self.system_prompt = system_prompt
        self.allowed_values = allowed_values

    def clean_data(self, df, column, clean_category_text, match_terms_not_exact, category_dict):
        """Clean and preprocess data for classification."""
        df['cleaned_noun'] = df[column].apply(clean_category_text)
        cleaned_category_dict = {clean_category_text(key): value for key, value in category_dict.items()}
        pattern_dict = {term: re.compile(r'\b' + re.escape(term) + r'\b', flags=re.IGNORECASE) for term in cleaned_category_dict}
        df[['matched_category', 'matched_noun']] = df['cleaned_noun'].apply(lambda noun: pd.Series(match_terms_not_exact(noun, pattern_dict, cleaned_category_dict, 'Unmatched')))
        df["matched_category_clean"] = df['matched_category'].apply(self.clean_header)
        return df

    def classify_unmatched(self, df_unmatched, predict_categories):
        """Classify unmatched categories using a fine-tuned model."""
        inputs = df_unmatched['short_description'].tolist()
        predictions = predict_categories(self.fine_tuned_model, inputs, self.system_prompt)
        df_unmatched['predictions_model'] = predictions
        df_unmatched["predictions_model_clean"] = df_unmatched['predictions_model'].apply(self.clean_header)
        df_unmatched = df_unmatched[df_unmatched["predictions_model_clean"].isin(self.allowed_values)]
        return predictions, df_unmatched

    @staticmethod
    def clean_header(text):
        """Format headers by removing extra spaces and capitalizing words."""
        return " ".join(text.split()).title()


class NounProcessor:
    def process_nouns(self, df, category_column):
        """Process and rename DataFrame columns for noun classification."""
        try:
            column_mapping = {'product_id': 'Product Id', category_column: 'predicted_category'}
            df.rename(columns=column_mapping, inplace=True)
            print("Columns renamed successfully for noun matching.")
            return df[['Product Id', 'predicted_category']]
        except KeyError as e:
            print(f"Error renaming columns: {e}")
            return None

