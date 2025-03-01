import os
import re

import kagglehub
import nltk
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

nltk.download("stopwords")
nltk.download("wordnet")

# Initializing
pyfile_path = os.path.abspath(__file__)
directory = os.path.dirname(pyfile_path)

# Download latest version
path = "/app/datasets/multimodal-mirex-emotion-dataset"


if not os.path.exists(path):
    print("Dataset not found. Downloading...")
    path = kagglehub.dataset_download("imsparsh/multimodal-mirex-emotion-dataset")
else:
    print("Dataset already exists.")

print("Data Loading done!")


# First verifications
songs_list = os.listdir(path + "/dataset/Audio")
print("number of songs in the dataset", len(songs_list))
with open(path + "/dataset/categories.txt", "r", encoding="utf-8") as f:
    emotions = f.readlines()
# Are there songs with no given emotions?
print("number of songs with emothions", len(emotions))
folders_list = os.listdir(path + "/dataset/Lyrics")
print("number of song with lyrics file", len(folders_list))

emotion_dict = {}
for i in range(1, len(emotions) + 1):
    emotion_dict[f"{i}".zfill(3) + ".txt"] = emotions[i - 1].strip("\n")

data = []
for file in folders_list:
    with open(path + f"/dataset/Lyrics/{file}", "r", encoding="utf-8") as f:
        lyrics = f.read()

    if file not in emotion_dict:
        print(f"Pas d'émotion pour la chanson {file}")
    else:
        data.append({"Lyrics": lyrics, "Emotion": emotion_dict[file]})

song_with_emotions = pd.DataFrame(data)
ouput_path = os.path.join(directory, "data_results/data_combined.csv")
song_with_emotions.to_csv(
    ouput_path, index=False, sep="|"
)  # Special char as a separator


# Cleaning the lyrics

# All in lower case
clean_lyrics = [song["Lyrics"].lower() for song in data]
# No special characters
clean_lyrics = [re.sub(r"[^a-zA-Z0-9]", " ", song) for song in clean_lyrics]
# No stopwords ("the", "a", "is"…)
stop_words = set(stopwords.words("english"))
clean_lyrics = [
    " ".join([word for word in song.split() if word not in stop_words])
    for song in clean_lyrics
]
# Lemmatisation/stemmatisation
lemmatizer = WordNetLemmatizer()
clean_lyrics = [
    " ".join(
        [lemmatizer.lemmatize(word) for word in song.split() if word not in stop_words]
    )
    for song in clean_lyrics
]
song_with_emotions["Cleaned lyrics"] = clean_lyrics
ouput_path = os.path.join(directory, "data_results/data_cleaned.csv")
song_with_emotions.drop(["Lyrics"], axis=1).to_csv(
    ouput_path, index=False, sep="|"
)  # Special char as a separator

print("Data Cleaning done!")


# Training a model (BERT)
# Other viable options are  WORD2VEC, FASTTEXT, TF-IDF
X = song_with_emotions["Cleaned lyrics"]
y = song_with_emotions["Emotion"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
model = BertModel.from_pretrained("distilbert-base-uncased")


def training_process(tokenizer, model, sentence):
    # Tokenize the data and convert into PyTorch vectors
    tokens_encoded = tokenizer(
        sentence, return_tensors="pt", truncation=True, max_length=512
    )

    # Get the embeddings
    with torch.no_grad():  # Deactivate the training of bert model
        outputs = model(**tokens_encoded)

    # Extract the embeddings
    last_hidden_state = outputs.last_hidden_state
    cls_embedding = last_hidden_state[:, 0, :]

    # Print the shape of the embeddings
    print("Shape of last hidden state:", last_hidden_state.shape)
    print("Shape of CLS embedding:", cls_embedding.shape)
    return cls_embedding


song_with_emotions["embeddings"] = [
    training_process(tokenizer, model, sentence)
    for sentence in song_with_emotions["Cleaned lyrics"]
]
print(song_with_emotions["embeddings"][0].shape)  # should be (1, 768)

# make it usable for torch (either a tensor ou un numpy)
song_with_emotions["embeddings"] = [
    embedding.numpy() for embedding in song_with_emotions["embeddings"]
]
label_encoder = preprocessing.LabelEncoder()
song_with_emotions["labels_emotion"] = label_encoder.fit_transform(
    song_with_emotions["Emotion"]
)

emotion_dict = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print("emotion & index:", emotion_dict)

ouput_path = os.path.join(directory, "data_results/data_with_embeddings.csv")
song_with_emotions.to_csv(ouput_path, index=False, sep="|")


### Training a model (LSTM)
# Splitting the data
X = song_with_emotions["embeddings"]
y = song_with_emotions["labels_emotion"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Make the training data usable as tensors
X_train = [torch.tensor(x, dtype=torch.float32) for x in X_train]
X_train = torch.stack(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

print("Checking the shape of X_train and y_train:", X_train.shape, y_train.shape)

# Hyperparameters
sequence_length = 1  # Taille de la séquence
feature_dim = 768  # Taille des embeddings
hidden_dim = 128  # Dimension de l'état caché (choisi arbitrairement)
batch_size = 32  # Nombre d'exemples par lot
number_emotions = len(set(y))  # Nombre d'émotions (classes de sortie)
num_epochs = 15  # Nombre d'epochs

# Création du modèle
lstm = nn.LSTM(
    input_size=feature_dim, hidden_size=hidden_dim, batch_first=True
)  # LSTM layer
fc = nn.Linear(hidden_dim, number_emotions)  # Fully connected layer
softmax = nn.Softmax(dim=-1)  # Softmax activation for classification
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=0.001)

# Add a dim for LSTM , it requires a (batch_size, sequence_length, feature_dim) format
train_dataset = TensorDataset(
    X_train, y_train
)  # TensorDataset : group les x, y in one object
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)  # DataLoader: load the data in batches efficiently

# Model training
for epoch in range(num_epochs):
    for batch in train_loader:
        X_batch, y_batch = batch

        optimizer.zero_grad()

        # Pass through LSTM
        lstm_out, (h_n, c_n) = lstm(X_batch)  # Get LSTM outputs
        last_hidden_state = h_n[-1]  # Take the last hidden state

        # Pass through the fully connected layer
        outputs = fc(last_hidden_state)

        # Apply Softmax for classification probabilities
        outputs = softmax(outputs)

        loss = loss_fn(outputs, y_batch)
        loss.backward()  # Backpropagation
        optimizer.step()  #  Update weights

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Model evaluation
model.eval()
X_test = [torch.tensor(x, dtype=torch.float32) for x in X_test]
X_test = torch.stack(X_test)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

with torch.no_grad():
    #  Pass through LSTM
    lstm_out, (h_n, c_n) = lstm(X_test)  # Get LSTM outputs
    last_hidden_state = h_n[-1]  # Take the last hidden state

    # Pass through the fully connected layer
    outputs = fc(last_hidden_state)

    # Apply Softmax for classification probabilities
    outputs = softmax(outputs)

    loss = loss_fn(outputs, y_test)
    predicted_labels = outputs.argmax(dim=-1)
    accuracy = (predicted_labels == y_test).sum() / len(y_test)

    print(f"Test Loss: {loss.item():.4f}")
    print(f"Test Accuracy: {accuracy.item():.4f}")
