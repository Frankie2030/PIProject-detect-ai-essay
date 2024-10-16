# %pip install scipy seaborn scikit-learn tensorflow tensorflow_datasets 
# %pip install pandas
# %pip install matplotlib
# %pip install numpy
# %pip install sklearn
# %pip install xgboost
# %pip install gensim
# %pip install nltk
# %pip install tqdm
# %pip install pyvi
# %pip install tensorflow
# %pip install torch

## Imports
from tqdm import tqdm
import os
import time
import re
import joblib
import torch
import logging
import nltk
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# ref: https://www.kaggle.com/code/eswarbabu88/toxic-comment-glove-logistic-regression
# need to use glove_model from above
# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ===========================================================================================
# Tạo class Logger
class MyLogger:
    def __init__(self, log_file='app.log'):
        self.log_file = log_file
        self._initialize_logger()

    def _initialize_logger(self):
        # Check if the log file already exists; if so, append to it
        if os.path.exists(self.log_file):
            file_mode = 'a'  # Append mode
        else:
            file_mode = 'w'  # Write mode (create new file)

        # Create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Set logging level to INFO

        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # Ensure the file handler logs INFO and above

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Ensure the console handler logs INFO and above

        # Set up the logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Remove existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_message(self, message):
        self.logger.info(message)

    def change_log_file(self, new_log_file):
        """Change the log file and reinitialize the logger."""
        self.log_file = new_log_file
        self._initialize_logger()  # Reinitialize logger with the new log file

# ============================================================================================
# danh sách các biến toàn cục
logger = MyLogger()

# Enable/Disable tokenizers parallelism to avoid the warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ============================================================================================
# Danh sách các hàm xử lý
# ============================================================================================
# Load the stop words
stop_words = set(stopwords.words('english'))

# Function to remove stop words
def remove_stop_words(text):
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Join the filtered words back into a string
    return ' '.join(filtered_words)

def preprocess_text(text):
    # Step 1: Remove URLs
    text = re.sub(r'http\S+|https?://\S+|www\.\S+', '', text)
    
    # Step 2: Remove text in square brackets
    text = re.sub(r'\[.*?\]', '', text)
    
    # Step 3: Remove angle brackets
    text = re.sub(r'<.*?>+', '', text)
    
    # Step 4: Remove newlines, tabs, carriage returns, form feeds, backspace characters
    text = re.sub(r'\n|\t|\r|\f|\b', '', text)
    
    # Step 5: Remove words that contain numbers
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Step 6: Remove any non-alphanumeric characters, then make lowercase
    text = re.sub(r'\W+', ' ', text).lower().strip()
    
    # Step 7: Tokenize the Vietnamese text using ViTokenizer
    text = ViTokenizer.tokenize(text)
    
    text = remove_stop_words(text)
    
    return text

def load_data(file_path):
    # Load the dataset
    df_ds = pd.read_csv(file_path)

    # First, split into train and test sets (80% train, 20% test)
    train_essays, test_essays = train_test_split(df_ds, test_size=0.2, random_state=42)

    # Then, split the train set into train and validation sets (67% train, 33% validation)
    train_essays, val_essays = train_test_split(train_essays, test_size=0.33, random_state=42)
    
    return df_ds, train_essays, test_essays, val_essays

def compute_metrics(preds, labels):
    # Convert probabilities to binary predictions
    binary_preds = (preds >= 0.5).astype(int)

    # Compute ROC AUC score
    auc = roc_auc_score(labels, preds)

    # Other metrics with zero_division set to handle undefined metrics
    accuracy = accuracy_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds, zero_division=0)  # Use zero_division=0 to avoid warnings
    recall = recall_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds, zero_division=0)  # Use zero_division=0 to avoid warnings

    return {"roc_auc": auc, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def compute_metrics_bert(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Get the predicted class by selecting the class with the highest probability
    preds = np.argmax(probs, axis=-1)
    
    # Compute ROC AUC score using probabilities of the positive class (class 1)
    if len(np.unique(labels)) > 1:  # Ensure there are both classes present for AUC computation
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = float('nan')  # AUC is not defined if only one class is present
    
    # Compute other metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    return {"roc_auc": auc, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def plot_tsne(model, num):
    labels = []
    tokens = []
    for word in model.key_to_index:
        if word not in stop_words:
            tokens.append(np.array(model[word]))
            labels.append(word)
    tsne = TSNE(perplexity = 40, n_components = 2, init = 'pca', n_iter = 2500, random_state = 23)
    data = tsne.fit_transform(np.array(tokens[:num]))
    x = []
    y = []
    for each in data:
        x.append(each[0])
        y.append(each[1])
    plt.figure(figsize = (10, 10))
    for i in range(num):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy = (x[i], y[i]),
                     xytext = (5,2),
                     textcoords = 'offset points',
                     ha = 'right',
                     va = 'bottom')
    plt.show()

# Assuming glove_model is already loaded in your environment
# Function to convert a sentence to a vector
def sent2vec(s, glove_model):
    words = str(s).lower()
    words = word_tokenize(words)  # This requires the 'punkt' tokenizer
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if w.isalpha()]  # Filter out non-alphabetic tokens
    M = []
    for w in words:
        try:
            M.append(glove_model[w])  # Lookup word in GloVe model
        except KeyError:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)  # Return a zero vector if no word embeddings are found
    return v / np.sqrt((v ** 2).sum())  # Normalize the vector

# ============================================================================================
def plot_training_validation_curves(train_metrics, val_metrics, save_path, file_name):
    # Create the output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['accuracy'], label='Train Accuracy')
    plt.plot(val_metrics['accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics['loss'], label='Train Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    # Save the plot to the specified directory with the specified file name
    plot_path = os.path.join(save_path, file_name)
    plt.savefig(plot_path)

    # Show the plot as well
    plt.show()

    print(f"Plot saved to: {plot_path}")

def test_model(X_test_embeddings, y_test, model_file_path):
    # Load the trained model
    model = joblib.load(model_file_path)
    
    # Predict probabilities
    preds_test = model.predict_proba(X_test_embeddings)[:, 1]
    
    # Compute test metrics
    test_loss = log_loss(y_test, preds_test)
    test_metrics = compute_metrics(preds_test, y_test)

    # Log test metrics
    logger.log_message("Testing Results:")
    logger.log_message(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, ")
    logger.log_message(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}, Precision: {test_metrics['precision']:.4f}, ")
    logger.log_message(f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # Print test metrics for console
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, "
          f"Test ROC AUC: {test_metrics['roc_auc']:.4f}, Precision: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # Return all test metrics
    return {"loss": test_loss, **test_metrics}

def test_linearsvc(X_test_embeddings, y_test, model_file_path):
    # Load the trained model
    model = joblib.load(model_file_path)
    
    # Predict decision function values (not probabilities) for the test set
    preds_test = model.decision_function(X_test_embeddings)

    # Since LinearSVC doesn't output probabilities, you can use the decision function output directly
    # Compute the test loss using log loss (note: log_loss usually expects probabilities, but here we use decision values)
    # You might want to threshold or normalize preds_test before passing it to log_loss
    test_loss = log_loss(y_test, preds_test, labels=np.unique(y_test))  # Ensure labels are passed to handle binary classification

    # Compute test metrics
    test_metrics = compute_metrics(preds_test, y_test)

    # Log test metrics
    logger.log_message("Testing Results:")
    logger.log_message(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, ")
    logger.log_message(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}, Precision: {test_metrics['precision']:.4f}, ")
    logger.log_message(f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # Print test metrics for console
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, "
          f"Test ROC AUC: {test_metrics['roc_auc']:.4f}, Precision: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # Return all test metrics
    return {"loss": test_loss, **test_metrics}

def test_rnn(X_test, y_test, model_file_path):
    """
    Function to test a trained RNN model using TensorFlow/Keras.
    It loads the model, performs predictions, and computes metrics.
    """
    # Load the trained model
    if os.path.exists(model_file_path):
        print("Loading model from file...")
        model = tf.keras.models.load_model(model_file_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_file_path}")

    # Create a TensorFlow dataset for testing
    batch_size = 32
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Predict logits (raw output before activation)
    preds_test_logits = model.predict(test_dataset)

    # Convert logits to probabilities using sigmoid since we are doing binary classification
    preds_test = tf.sigmoid(preds_test_logits).numpy().flatten()

    # Compute test loss using log loss (cross-entropy)
    test_loss = log_loss(y_test, preds_test)

    # Compute test metrics (precision, recall, F1, etc.)
    test_metrics = compute_metrics(preds_test, y_test)

    # Log test metrics
    logger.log_message("Testing Results:")
    logger.log_message(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, ")
    logger.log_message(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}, Precision: {test_metrics['precision']:.4f}, ")
    logger.log_message(f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

    # Print test metrics to console
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, "
          f"Test ROC AUC: {test_metrics['roc_auc']:.4f}, Precision: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

    # Return test loss and all test metrics
    return {"loss": test_loss, **test_metrics}

def test_distilbert(X_test, model_file_path):
    """
    Function to test a DistilBERT model on the test dataset and print metrics.
    """
    # Load the trained model and tokenizer
    if os.path.exists(model_file_path):
        print("Loading model and tokenizer from file...")
        model = AutoModelForSequenceClassification.from_pretrained(model_file_path)
        tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_file_path}")

    # Create the test dataset
    test_dataset = Dataset.from_pandas(X_test)

    # Tokenize the test dataset
    tokenized_test = test_dataset.map(lambda examples: tokenizer(examples['processed_text_swr'], max_length=128, padding=True, truncation=True), batched=True)

    # Initialize the Trainer (no training arguments needed for testing)
    trainer = Trainer(model=model, tokenizer=tokenizer)

    # Perform predictions on the test set
    predictions = trainer.predict(tokenized_test)
    preds_test = predictions.predictions.argmax(axis=-1)

    # Compute metrics for the test set using the logits and labels from predictions
    test_metrics = compute_metrics(predictions.predictions, predictions.label_ids)

    # Log and print the test metrics
    logger.log_message("Testing Results:")
    logger.log_message(f"Test Accuracy: {test_metrics['accuracy']:.4f}, Test ROC AUC: {test_metrics['roc_auc']:.4f}, ")
    logger.log_message(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, Test ROC AUC: {test_metrics['roc_auc']:.4f}, "
          f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

    return test_metrics


# ============================================================================================
def shuffle_and_split(X, y, batch_size):
    """Shuffle the data and split into batches."""
    indices = np.random.permutation(len(y))  # Shuffle the indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Divide into batches
    num_batches = len(y) // batch_size
    X_batches = np.array_split(X_shuffled, num_batches)
    y_batches = np.array_split(y_shuffled, num_batches)
    
    return X_batches, y_batches

#######
def train_logistic_regression(X_train_embeddings, y_train, X_val_embeddings, y_val, model_file_path, out_base_path, n_epochs=20, batch_size=32):
    logger.log_message(f"Training a Logistic Regression model for {n_epochs} epochs...")

    if os.path.exists(model_file_path):
        print("Loading model from file...")
        model = joblib.load(model_file_path)
    else:
        print("Model not found. Training a new model...")
        model = LogisticRegression(max_iter=246, C=0.01, class_weight='balanced')

        train_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}
        val_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}

        # Loop over epochs
        for epoch in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):
            logger.log_message(f"Epoch {epoch + 1}/{n_epochs} - Shuffling and batching data")

            # Shuffle and split data into batches
            X_train_batches, y_train_batches = shuffle_and_split(X_train_embeddings, y_train, batch_size)

            # Train model on each batch
            for X_batch, y_batch in zip(X_train_batches, y_train_batches):
                model.fit(X_batch, y_batch)

            # After training on batches, validate
            preds_train = model.predict_proba(X_train_embeddings)[:, 1]
            preds_val = model.predict_proba(X_val_embeddings)[:, 1]

            # Compute training metrics
            train_loss = log_loss(y_train, preds_train)
            train_metrics_epoch = compute_metrics(preds_train, y_train)
            train_metrics['loss'].append(train_loss)
            for key, value in train_metrics_epoch.items():
                train_metrics[key].append(value)

            # Compute validation metrics
            val_loss = log_loss(y_val, preds_val)
            val_metrics_epoch = compute_metrics(preds_val, y_val)
            val_metrics['loss'].append(val_loss)
            for key, value in val_metrics_epoch.items():
                val_metrics[key].append(value)

            # Print metrics for the epoch
            # logger.log_message(f"Epoch {epoch + 1}/{n_epochs}")
            logger.log_message(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, ")
            logger.log_message(f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, ")
            logger.log_message(f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")
            
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, "
                  f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, "
                  f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics_epoch['accuracy']:.4f}, "
                  f"Val ROC AUC: {val_metrics_epoch['roc_auc']:.4f}, Precision: {val_metrics_epoch['precision']:.4f}, "
                  f"Recall: {val_metrics_epoch['recall']:.4f}, F1: {val_metrics_epoch['f1']:.4f}")

        # Save the trained model
        joblib.dump(model, model_file_path)
        print(f"Model saved to {model_file_path}")

        # Plot accuracy and loss curves
        plot_training_validation_curves(train_metrics, val_metrics, out_base_path, "LogisticRegressionAnalysis.png")

    return model

#######
def train_xgboost(X_train_embeddings, y_train, X_val_embeddings, y_val, model_file_path, out_base_path, n_epochs=20, batch_size=32):
    logger.log_message(f"Training an XGBoost model for {n_epochs} epochs...")

    if os.path.exists(model_file_path):
        print("Loading model from file...")
        model = joblib.load(model_file_path)
    else:
        print("Model not found. Training a new model...")
        # model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.01, n_estimators=100)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.01, n_estimators=100)

        train_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}
        val_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}

        # Loop over epochs
        for epoch in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):
            logger.log_message(f"Epoch {epoch + 1}/{n_epochs} - Shuffling and batching data")

            # Shuffle and split data into batches
            X_train_batches, y_train_batches = shuffle_and_split(X_train_embeddings, y_train, batch_size)

            # Train model on each batch
            for X_batch, y_batch in zip(X_train_batches, y_train_batches):
                model.fit(X_batch, y_batch)

            # After training on batches, validate
            preds_train = model.predict_proba(X_train_embeddings)[:, 1]
            preds_val = model.predict_proba(X_val_embeddings)[:, 1]

            # Compute training metrics
            train_loss = log_loss(y_train, preds_train)
            train_metrics_epoch = compute_metrics(preds_train, y_train)
            train_metrics['loss'].append(train_loss)
            for key, value in train_metrics_epoch.items():
                train_metrics[key].append(value)

            # Compute validation metrics
            val_loss = log_loss(y_val, preds_val)
            val_metrics_epoch = compute_metrics(preds_val, y_val)
            val_metrics['loss'].append(val_loss)
            for key, value in val_metrics_epoch.items():
                val_metrics[key].append(value)

            # Print metrics for the epoch
            # logger.log_message(f"Epoch {epoch + 1}/{n_epochs}")
            logger.log_message(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, ")
            logger.log_message(f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, ")
            logger.log_message(f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")

            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, "
                  f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, "
                  f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics_epoch['accuracy']:.4f}, "
                  f"Val ROC AUC: {val_metrics_epoch['roc_auc']:.4f}, Precision: {val_metrics_epoch['precision']:.4f}, "
                  f"Recall: {val_metrics_epoch['recall']:.4f}, F1: {val_metrics_epoch['f1']:.4f}")

        # Save the trained model
        joblib.dump(model, model_file_path)
        print(f"Model saved to {model_file_path}")

        # Plot accuracy and loss curves
        plot_training_validation_curves(train_metrics, val_metrics, out_base_path, "xgboost_training_validation_curves.jpg")

    return model

#######
def train_random_forest(X_train_embeddings, y_train, X_val_embeddings, y_val, model_file_path, out_base_path, n_epochs=20, batch_size=32):
    logger.log_message(f"Training a Random Forest model for {n_epochs} epochs...")

    if os.path.exists(model_file_path):
        print("Loading model from file...")
        model = joblib.load(model_file_path)
    else:
        print("Model not found. Training a new model...")
        model = RandomForestClassifier(n_estimators=24, max_depth=10, random_state=42)

        train_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}
        val_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}

        # Loop over epochs
        for epoch in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):
            logger.log_message(f"Epoch {epoch + 1}/{n_epochs} - Shuffling and batching data")

            # Shuffle and split data into batches
            X_train_batches, y_train_batches = shuffle_and_split(X_train_embeddings, y_train, batch_size)

            # Train model on each batch
            for X_batch, y_batch in zip(X_train_batches, y_train_batches):
                start = time.time()
                model.fit(X_batch, y_batch)
                # print(f"Model fit time: {time.time() - start} seconds")

            # After training on batches, validate
            preds_train = model.predict_proba(X_train_embeddings)[:, 1]
            preds_val = model.predict_proba(X_val_embeddings)[:, 1]

            # Compute training metrics
            train_loss = log_loss(y_train, preds_train)
            train_metrics_epoch = compute_metrics(preds_train, y_train)
            train_metrics['loss'].append(train_loss)
            for key, value in train_metrics_epoch.items():
                train_metrics[key].append(value)

            # Compute validation metrics
            val_loss = log_loss(y_val, preds_val)
            val_metrics_epoch = compute_metrics(preds_val, y_val)
            val_metrics['loss'].append(val_loss)
            for key, value in val_metrics_epoch.items():
                val_metrics[key].append(value)

            # Print metrics for the epoch
            # logger.log_message(f"Epoch {epoch + 1}/{n_epochs}")
            logger.log_message(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, ")
            logger.log_message(f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, ")
            logger.log_message(f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")

            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, "
                  f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, "
                  f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics_epoch['accuracy']:.4f}, "
                  f"Val ROC AUC: {val_metrics_epoch['roc_auc']:.4f}, Precision: {val_metrics_epoch['precision']:.4f}, "
                  f"Recall: {val_metrics_epoch['recall']:.4f}, F1: {val_metrics_epoch['f1']:.4f}")

        # Save the trained model
        joblib.dump(model, model_file_path)
        print(f"Model saved to {model_file_path}")

        # Plot accuracy and loss curves
        plot_training_validation_curves(train_metrics, val_metrics, out_base_path, "RF_training_validation_curves.jpg")

    return model

#######
def train_linear_svc(X_train_embeddings, y_train, X_val_embeddings, y_val, model_file_path, out_base_path, n_epochs=20, batch_size=32):
    logger.log_message(f"Training a LinearSVC model for {n_epochs} epochs...")

    if os.path.exists(model_file_path):
        print("Loading model from file...")
        model = joblib.load(model_file_path)
    else:
        print("Model not found. Training a new model...")
        # Instantiate the LinearSVC model
        model = LinearSVC(max_iter=246, class_weight='balanced', tol=1e-4, C=1.0)

        train_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}
        val_metrics = {'accuracy': [], 'loss': [], 'roc_auc': [], 'precision': [], 'recall': [], 'f1': []}

        # Loop over epochs
        for epoch in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):
            logger.log_message(f"Epoch {epoch + 1}/{n_epochs} - Shuffling and batching data")

            # Shuffle and split data into batches
            X_train_batches, y_train_batches = shuffle_and_split(X_train_embeddings, y_train, batch_size)

            # Train model on each batch
            for X_batch, y_batch in zip(X_train_batches, y_train_batches):
                model.fit(X_batch, y_batch)

            # After training on batches, validate
            preds_train = model.decision_function(X_train_embeddings)
            preds_val = model.decision_function(X_val_embeddings)

            # Compute training metrics
            train_loss = log_loss(y_train, preds_train)
            train_metrics_epoch = compute_metrics(preds_train, y_train)
            train_metrics['loss'].append(train_loss)
            for key, value in train_metrics_epoch.items():
                train_metrics[key].append(value)

            # Compute validation metrics
            val_loss = log_loss(y_val, preds_val)
            val_metrics_epoch = compute_metrics(preds_val, y_val)
            val_metrics['loss'].append(val_loss)
            for key, value in val_metrics_epoch.items():
                val_metrics[key].append(value)

            # Print metrics for the epoch
            logger.log_message(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, ")
            logger.log_message(f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, ")
            logger.log_message(f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")

            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics_epoch['accuracy']:.4f}, "
                  f"Train ROC AUC: {train_metrics_epoch['roc_auc']:.4f}, Precision: {train_metrics_epoch['precision']:.4f}, "
                  f"Recall: {train_metrics_epoch['recall']:.4f}, F1: {train_metrics_epoch['f1']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics_epoch['accuracy']:.4f}, "
                  f"Val ROC AUC: {val_metrics_epoch['roc_auc']:.4f}, Precision: {val_metrics_epoch['precision']:.4f}, "
                  f"Recall: {val_metrics_epoch['recall']:.4f}, F1: {val_metrics_epoch['f1']:.4f}")

        # Save the trained model
        joblib.dump(model, model_file_path)
        print(f"Model saved to {model_file_path}")

        # Plot accuracy and loss curves
        plot_training_validation_curves(train_metrics, val_metrics, out_base_path, "LinearSVC_training_validation_curves.png")

    return model

#######
def train_rnn(X_train, y_train, X_val, y_val, model_file_path, plot_file_path, n_epochs=20, batch_size=32):
    """
    Function to train a Bidirectional LSTM RNN using TensorFlow/Keras.
    It checks if the model already exists, otherwise it trains and saves the model.
    """
    # Create TensorFlow datasets from the input data
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    # Check if the model already exists
    if os.path.exists(model_file_path):
        print("Loading model from file...")
        model = tf.keras.models.load_model(model_file_path)
    else:
        print("Model not found. Training a new model...")

        # Text vectorization layer
        VOCAB_SIZE = 1000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder.adapt(train_dataset.map(lambda text, label: text))

        # Define the Bidirectional LSTM RNN model
        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

        # Train the model
        history = model.fit(train_dataset, epochs=n_epochs, validation_data=val_dataset)

        # Save the model
        model.save(model_file_path)
        print(f"Model saved to {model_file_path}")

        # Plot training and validation accuracy and loss
        def plot_graphs(history, metric):
            plt.plot(history.history[metric])
            plt.plot(history.history['val_' + metric], '')
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend([metric, 'val_' + metric])

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plot_graphs(history, 'accuracy')
        plt.ylim(None, 1)
        plt.subplot(1, 2, 2)
        plot_graphs(history, 'loss')
        plt.ylim(0, None)

        # Save the plot
        plt.savefig(plot_file_path)
        print(f"Training plot saved to {plot_file_path}")
        plt.show()

    return model

#######
def preprocess_function(examples, tokenizer):
    return tokenizer(examples["processed_text_swr"], max_length=128, padding=True, truncation=True)

def plot_graphs(log_history, metric):
    epochs = [entry['epoch'] for entry in log_history if metric in entry]
    metric_values = [entry[metric] for entry in log_history if metric in entry]
    plt.plot(epochs, metric_values)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric])

def train_distilbert(df_ds, model_file_path, image_file_path, n_epochs=20):
    """
    Function to train a DistilBERT model. 
    It checks if the model already exists, otherwise it trains and saves the model.
    """
    # Check if model exists
    if os.path.exists(model_file_path):
        print("Loading model from file...")
        model = AutoModelForSequenceClassification.from_pretrained(model_file_path)
    else:
        print("Model not found. Training a new model...")

        # Split dataset
        train_essays, test_essays = train_test_split(df_ds, test_size=0.2, random_state=42)
        train_essays, val_essays = train_test_split(train_essays, test_size=0.33, random_state=42)

        # Tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        # Create dataset
        train_essay_dataset = Dataset.from_pandas(train_essays)
        val_essay_dataset = Dataset.from_pandas(val_essays)
        
        # Tokenize datasets, passing the tokenizer as an argument to the map function
        tokenized_train_essays = train_essay_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
        tokenized_val_essays = val_essay_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="/kaggle/working/",
            learning_rate=2e-5,
            num_train_epochs=n_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to='none'
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_essays,
            eval_dataset=tokenized_val_essays,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_bert
        )

        # Train the model
        trainer.train()

        # Save the trained model
        model.save_pretrained(model_file_path)  # Updated to Hugging Face save method
        tokenizer.save_pretrained(model_file_path)  # Save tokenizer too
        print(f"Model saved to {model_file_path}")

        # Plot accuracy and loss graphs
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plot_graphs(trainer.state.log_history, 'eval_accuracy')
        plt.ylim(None, 1)
        plt.subplot(1, 2, 2)
        plot_graphs(trainer.state.log_history, 'eval_loss')
        plt.ylim(0, None)

        # Save the plot
        plt.savefig(image_file_path)
        plt.show()
        print(f"Training plot saved as {image_file_path}")

    return model


# ============================================================================================



# ============================================================================================
# Hàm chính
# ============================================================================================
def main():
    # Khởi tạo số luồng xử lý song song
    # max_workers = 1 
    
    # kaggle 
    in_base_path = r"/kaggle/input/dath-pdz/"
    out_base_path = r"/kaggle/working/"
    
    # in_base_path = r"E:\2_LEARNING_BKU\2_File_2\K22_HK241\CO3101_Do_an_Tri_tue_nhan_tao\Main\Dataset"
    # out_base_path = r"E:\2_LEARNING_BKU\2_File_2\K22_HK241\CO3101_Do_an_Tri_tue_nhan_tao\Output"   # đường dẫn gốc tới folder
    
    # Fix the file path by adding the missing backslash or using os.path.join
    file_name = os.path.join(in_base_path, 'final_dataset_v1_afternb1.csv')  # Correct file path
    
    # Bắt đầu theo dõi thời gian
    t_start_time = time.time()
    
    # Load and preprocess data
    df_ds, train_essays, test_essays, val_essays = load_data(file_name)
    
    # Check the size of each set
    print(f'Full set size: {len(df_ds)}')
    print(f'Training set size: {len(train_essays)}')
    print(f'Validation set size: {len(val_essays)}')
    print(f'Test set size: {len(test_essays)}')

    # ============================================================================================
    # Load the glove model
    word2vec_output_file = get_tmpfile(r"/kaggle/input/pdz-dath-ds/output_w2v.txt")
    # word2vec_output_file = get_tmpfile(r"E:\2_LEARNING_BKU\2_File_2\K22_HK241\CO3101_Do_an_Tri_tue_nhan_tao\Main\Dataset\output_w2v.txt")
    glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    # Prepare train and validation embeddings
    X_train = train_essays['processed_text_swr'].tolist()
    X_val = val_essays['processed_text_swr'].tolist()
    y_train = train_essays['label'].values
    y_val = val_essays['label'].values
    
    # Prepare test data
    X_test = test_essays['processed_text_swr'].tolist()
    y_test = test_essays['label'].values
    
    # Embedding these information dataset
    X_train_embeddings = np.array([sent2vec(sent, glove_model) for sent in X_train])
    X_val_embeddings = np.array([sent2vec(sent, glove_model) for sent in X_val])
    
    X_test_embeddings = np.array([sent2vec(sent, glove_model) for sent in X_test])
    # ============================================================================================
    # ============================================================================================
    # ============================================================================================
    # ============================================================================================
    # ============================================================================================
    # ============================================================================================

    distilbert_model_path = os.path.join(out_base_path, 'distilbert_model')  # Use folder path without file extension
    distilbert_plot_file_path = os.path.join(out_base_path, 'distilbert_training_plot.png')

    print("Training DistilBERT Model")
    train_distilbert(df_ds, distilbert_model_path, distilbert_plot_file_path, n_epochs=10)

    # Test the DistilBERT model on the test set
    print("Testing DistilBERT Model")
    test_distilbert(test_essays, distilbert_model_path)

    # ============================================================================================
    
    # Kết thúc theo dõi thời gian
    t_end_time = time.time()
    t_processing_time = t_end_time - t_start_time

    # Convert minutes to hours and minutes
    t_hours = int(t_processing_time // 3600)  # Lấy số giờ
    t_minutes = int((t_processing_time % 3600) // 60)  # Lấy số phút
    t_seconds = int(t_processing_time % 60)  # Lấy số giây

    logger.log_message(f"Finished processing (total) in {t_hours} hours, {t_minutes} minutes, {t_seconds} seconds")       

if __name__ == "__main__":  
    main()


