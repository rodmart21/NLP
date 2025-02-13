import os
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .constants import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# New imports
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

import spacy
nlp = spacy.load("en_core_web_md")

from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

## Evaluation

def plot_learning_curve(estimator, title, X, y):                 # Evaluate a model's performance and scalability as the size of the training data increases
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features).

    y : array-like, shape (n_samples) or (n_samples, n_features).

    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_title(title)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(        # Outputs: train size in every step. Train score in ebvery training size. Fit_times: time to train the model in each training size
        estimator, X, y, cv=5, n_jobs=4, train_sizes=np.linspace(0.1, 1, 6), return_times=True)
    
    # Compute metrics: across 5 folds.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
    axes[0].plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt   # Optimized: more arguments in the function. 3 plots into separate helper functions (every olot in a function)->better readability. Seaborn also improves.


def plot_confusion_matrix(            # Could be optimized using seaborn here. Adding color or normalization as arguments for the function.
    cm, classes,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # Normalization: for having sum up to 1.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)



# Cleaning function:

def clean_text(text):
    """ Full text preprocessing function with stopwords removal, lemmatization, and stemming """
    if pd.isna(text) or not isinstance(text, str):
        return "", "", "", ""  # Return empty strings for all variations
    
    # 1. Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize words
    words = text.split()

    # 5. Expand abbreviations
    words = [abbreviations.get(word, word) for word in words]

    # 6. Remove stopwords and business-specific words
    words = [word for word in words if word not in stop_words and word not in company_stopwords]

    # 7. Lemmatized text without stopwords
    lemmatized_no_stopwords = ' '.join([lemmatizer.lemmatize(word) for word in words])

    # 8. Stemmed text without stopwords
    stemmed_no_stopwords = ' '.join([stemmer.stem(word) for word in words])

    # 9. Lemmatized text with stopwords (original cleaned version)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return lemmatized_text, lemmatized_no_stopwords, stemmed_no_stopwords


def tfidf_params(X_train, X_val, y_train, y_val, param_grid, verbose=False):  # Iterates through a list of hyperparameters. Simple LogReg model for best hyperparameters finding
    """
    Preprocesses and standardizes text by removing metadata, special characters, and stopwords,
    then generates multiple processed versions with lemmatization and stemming.

    Parameters:
    ----------
    text : str
        Input text string to preprocess.

    Returns:
    -------
    cleaned_text : str
        Cleaned text without additional processing.
    cleaned_no_stopwords : str
        Text without stopwords.
    cleaned_lemmatized : str
        Text lemmatized and without stopwords.
    cleaned_stemmed : str
        Text stemmed and without stopwords.
    """
    best_score = 0
    best_params = None
    best_vectorizer = None

    for params in ParameterGrid(param_grid):
        if verbose:
            print(f"Testing parameters: {params}")

        vectorizer = TfidfVectorizer(**params)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_tfidf, y_train)
        
        score = f1_score(y_val, clf.predict(X_val_tfidf), average='weighted')  

        # Why use "macro" instead of "weighted"?.  For not balanced cases, would be better using macro
        # Macro F1-score calculates the F1-score for each class individually and then takes the unweighted average.
        # Weighted F1-score accounts for class distribution, which is useful for imbalanced datasets, but unnecessary in your case.
        
        if score > best_score:
            best_score = score
            best_params = params
            best_vectorizer = vectorizer

        if verbose:
            print(f"Score for parameters {params}: {score:.4f}")

    X_train_tfidf = best_vectorizer.fit_transform(X_train)
    X_val_tfidf = best_vectorizer.transform(X_val)
    print("Best TF-IDF params:", best_params)
    print("Best F1-Score:", best_score)

    return best_vectorizer, X_train_tfidf, X_val_tfidf, best_params


# Tfidf with Grid Search using K-Folds and including validation

def tfidf_grid_search(X_train, X_val, y_train, y_val, param_grid, cv=5, scoring='f1_macro', verbose=False):
    """
    Uses GridSearchCV to find the best TF-IDF parameters and applies k-fold cross-validation.
    
    Parameters:
    ----------
    X_train, X_val : list
        Text data for training and validation.
    y_train, y_val : list
        Labels for training and validation.
    param_grid : dict
        Parameter grid for TF-IDF vectorization.
    cv : int, optional (default=5)
        Number of folds for cross-validation.
    scoring : str, optional (default='f1_macro')
        Scoring metric for model evaluation.
    verbose : bool, optional (default=False)
        If True, prints progress.

    Returns:
    --------
    best_vectorizer : TfidfVectorizer
        Best TF-IDF vectorizer found.
    X_train_tfidf, X_val_tfidf : sparse matrix
        Transformed training and validation data.
    best_params : dict
        Best parameters found in grid search.
    """

    # Define a pipeline with TF-IDF vectorization and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  
        ('clf', LogisticRegression(max_iter=1000))  
    ])

    # GridSearchCV to optimize TF-IDF hyperparameters
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={'tfidf__' + key: values for key, values in param_grid.items()},
        scoring=make_scorer(f1_score, average=scoring),
        cv=cv,
        verbose=verbose,
        n_jobs=-1
    )

    # Fit GridSearchCV on training data
    grid_search.fit(X_train, y_train)

    # Get the best TF-IDF vectorizer
    best_vectorizer = grid_search.best_estimator_.named_steps['tfidf']
    
    # Transform training and validation data using the best vectorizer
    X_train_tfidf = best_vectorizer.fit_transform(X_train)
    X_val_tfidf = best_vectorizer.transform(X_val)

    # Print best parameters and score
    print("Best TF-IDF params:", grid_search.best_params_)
    print("Best F1-Score:", grid_search.best_score_)

    return best_vectorizer, X_train_tfidf, X_val_tfidf, grid_search.best_params_

    

def train_logistic_regression(X_train_tfidf, y_train, param_grid, verbose=1):  # Grid Search. cv=5, for cross validation, make sure model is generalized. 
    # Splitting the dataset multiple times instead of a single train-test split. 
    # No actually validation data, just splitting the training one.
    """
    Train a Logistic Regression model using GridSearchCV with the given TF-IDF parameters.

    Parameters:
    ----------
    X_train_tfidf : sparse matrix
        TF-IDF transformed training data.
    y_train : array-like
        Labels for the training data.
    param_grid : dict
        Parameter grid for Logistic Regression hyperparameter tuning.
    verbose : int, optional (default=1)
        Level of verbosity during GridSearchCV execution.

    Returns:
    --------
    best_model : GridSearchCV
        Fitted GridSearchCV object with the best Logistic Regression model.
    best_params_logreg : dict
        Best parameters for Logistic Regression found during the grid search.
    """
    best_model = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        scoring='f1_weighted',
        cv=5,
        verbose=verbose)

    best_model.fit(X_train_tfidf, y_train)
    best_params_logreg = best_model.best_params_

    if verbose:
        print("Best Logistic Regression Params:", best_params_logreg)

    return best_model, best_params_logreg


def train_random_forest(X_train, y_train, param_grid, verbose=1):    # Actually same function, could have taken both of them together. 
    """
    Train a Random Forest model using GridSearchCV with the given parameter grid.

    Parameters:
    ----------
    X_train : array-like
        Training data (e.g., embeddings).
    y_train : array-like
        Labels for the training data.
    param_grid : dict
        Parameter grid for Random Forest hyperparameter tuning.
    verbose : int, optional (default=1)
        Level of verbosity during GridSearchCV execution.

    Returns:
    --------
    best_model : GridSearchCV
        Fitted GridSearchCV object with the best Random Forest model.
    best_params_rf : dict
        Best parameters for Random Forest found during the grid search.
    """
    best_model = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        scoring='f1_weighted',
        cv=5,
        verbose=verbose)

    best_model.fit(X_train, y_train)
    best_params_rf = best_model.best_params_

    if verbose:
        print("Best Random Forest Params:", best_params_rf)

    return best_model, best_params_rf



def get_spacy_embeddings(texts):      
    """
    Generate embeddings using SpaCy.
    
    Args:
        texts (list of str): Input texts to vectorize.
        
    Returns:
        np.ndarray: Document embeddings for all input texts.
    """
    embeddings = []
    for doc in nlp.pipe(texts, disable=["parser", "ner"]):   # Processes multiple texts in batch mode. Disables unnecessary components (name entity recognition and parser) 
        # Parser: identifies grammatical relationships between words. Compatationally expensive and unnecessary if only extracting embeddings.
        embeddings.append(doc.vector)  
    return np.array(embeddings)


def get_transformer_embeddings(model, texts):   
    """
    Generate embeddings using a SentenceTransformer model.
    
    Args:
        texts (list of str): Input texts to vectorize.
        
    Returns:
        np.ndarray: Document embeddings for all input texts.
    """
    embeddings = model.encode(texts, show_progress_bar=True)   
    return embeddings   # Array where each row is a sentence embedding. 


def evaluate_model_new(X_train_tfidf, y_train, y_test, y_test_pred, best_model):    
    """
    Evaluates the model's performance using predicted and true labels and generates relevant metrics and visualizations.

    Inputs:
        X_train_tfidf (sparse matrix): TF-IDF representation of the training data.
        y_train (array-like): Training labels for the training data.
        y_test (array-like): True labels for the test data.
        y_test_pred (array-like): Predicted labels for the test data.
        best_model (GridSearchCV): Trained GridSearchCV object containing the best model and parameters.

    Outputs:
        None: Prints accuracy, F1-score, classification report, and plots confusion matrix and learning curves.
    """

    ac_score = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"(accuracy, f1) are: ({ac_score:.6f}, {f1:.6f})")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 7))
    plot_confusion_matrix(cnf_matrix, classes=np.unique(y_test), title="Confusion Matrix")
    plt.show()
    
    plot_learning_curve(best_model.best_estimator_, "Learning Curves", X_train_tfidf, y_train)
    plt.show()



def evaluate_model_smote(X_train_resampled, y_train_resampled, best_params, y_test, y_test_pred, best_model): 
    """
    Evaluates the model's performance using predicted and true labels and generates relevant metrics and visualizations.

    Inputs:
        X_train_tfidf (sparse matrix): TF-IDF representation of the training data.
        y_train (array-like): Training labels for the training data.
        y_test (array-like): True labels for the test data.
        y_test_pred (array-like): Predicted labels for the test data.
        best_model (GridSearchCV): Trained GridSearchCV object containing the best model and parameters.

    Outputs:
        None: Prints accuracy, F1-score, classification report, and plots confusion matrix and learning curves.
    """

    ac_score = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"(accuracy, f1) are: ({ac_score:.6f}, {f1:.6f})")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 7))
    plot_confusion_matrix(cnf_matrix, classes=np.unique(y_test), title="Confusion Matrix")
    plt.show()
    
    final_model = LogisticRegression(
        solver=best_params['logreg__solver'],
        C=best_params['logreg__C'],
        penalty=best_params['logreg__penalty'],
        class_weight=best_params['logreg__class_weight'],
        max_iter=100)
    final_model.fit(X_train_resampled, y_train_resampled)

    plot_learning_curve(final_model, "Learning Curves", X_train_resampled, y_train_resampled)
    plt.show()



def train_logistic_regression_with_smote(X_train, y_train, param_grid, smote_grid, verbose=1):
    """
    Train a Logistic Regression model using GridSearchCV with integrated SMOTE.

    Parameters:
    ----------
    X_train : sparse matrix
        Training data.
    y_train : array-like
        Labels for the training data.
    param_grid : dict
        Parameter grid for Logistic Regression hyperparameter tuning.
    smote_grid : dict
        Grid for SMOTE sampling strategies.
    verbose : int, optional (default=1)
        Level of verbosity during GridSearchCV execution.

    Returns:
    --------
    best_model : GridSearchCV
        Fitted GridSearchCV object with the best Logistic Regression model.
    best_params : dict
        Best parameters found during the grid search, including SMOTE settings.
    X_train_resampled : array-like
        Resampled training data after applying SMOTE.
    y_train_resampled : array-like
        Resampled training labels after applying SMOTE.
    """
    X_train_resampled, y_train_resampled = None, None

    pipeline = Pipeline([('smote', SMOTE(random_state=42, k_neighbors=2)),  ('logreg', LogisticRegression(max_iter=1000))])
    
    combined_param_grid = {
        'smote__sampling_strategy': smote_grid['sampling_strategy'],
        'logreg__solver': param_grid['solver'],
        'logreg__C': param_grid['C'],
        'logreg__penalty': param_grid['penalty'],
        'logreg__class_weight': param_grid['class_weight'],}
    
    best_model = GridSearchCV(
        pipeline,
        combined_param_grid,
        scoring='f1_weighted',
        cv=5,
        verbose=verbose)
    
    best_model.fit(X_train, y_train)
    best_params = best_model.best_params_

    smote_step = best_model.best_estimator_.named_steps['smote']
    X_train_resampled, y_train_resampled = smote_step.fit_resample(X_train, y_train)
    if verbose:
        print("Best Parameters (with SMOTE):", best_params)
        print("Resampled Training Data Shape:", X_train_resampled.shape)
        print("Resampled Training Labels Distribution:", Counter(y_train_resampled))

    return best_model, best_params, X_train_resampled, y_train_resampled