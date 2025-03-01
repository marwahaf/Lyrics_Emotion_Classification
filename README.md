# Music Emotion Classification using BERT and LSTM

## Overview

This project focuses on classifying emotions conveyed by songs based on their lyrics. It utilizes the **Multimodal MIREX Emotion Dataset**, which includes both audio features and lyrics, each tagged with a corresponding emotion. The project uses **BERT** embeddings for text representation and an **LSTM** (Long Short-Term Memory) network for emotion classification.

---

## Requirements

You have two options for running this project: using Docker (recommended) or manually setting up the environment locally.

### 1. Using Docker (Recommended)

Docker simplifies the setup process by automatically installing all necessary dependencies in an isolated environment.

- **Build the Docker image**:

   ```bash
   docker build -t music-emotion-classification .
   ```

- **Run the Docker container**:

   ```bash
   docker run --rm -v $(pwd):/app music-emotion-classification
   ```

This will set up the environment, download any missing dependencies, and run the script.

### 2. Without Docker (Local Setup)

If you prefer to run the project locally, follow these steps:

- **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

- **Install Dependencies**:

   Use `pip` to install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

- **Run the Script**:

   To start training the model, execute:

   ```bash
   python main.py
   ```

---

## Dataset

The project uses the **Multimodal MIREX Emotion Dataset**, which includes:

- **Audio files**: Song data in MP3 format.
- **Lyrics files**: Text files containing the lyrics of each song.
- **Emotion labels**: Each song is associated with a specific emotion label.

The dataset can be downloaded automatically via `kagglehub` if not already available locally.

### Folder Structure

```
/dataset
    /Audio           # Audio files of songs
    /Lyrics          # Lyrics of the songs in text files
    /categories.txt  # Emotion categories for the songs
```

---

## Workflow

### 1. Data Loading and Verification

The first step of the script verifies if the required dataset is available locally. If not, it will download it using `kagglehub`.

### 2. Data Cleaning and Preprocessing

The song lyrics undergo the following preprocessing steps:

- **Lowercasing**: All lyrics are converted to lowercase.
- **Cleaning**: Special characters are removed.
- **Stopword Removal**: Using NLTK's stopword list to remove common words that don't add value.
- **Lemmatization**: The words are lemmatized using NLTK's `WordNetLemmatizer`.

The cleaned lyrics are stored in a DataFrame, alongside their respective emotion labels.

### 3. Generating BERT Embeddings

For each song's lyrics, we use **DistilBERT** (a lighter version of BERT) to extract embeddings, which will serve as the input features for the LSTM model.

### 4. Model Training: LSTM

An **LSTM (Long Short-Term Memory)** model is trained using the BERT embeddings. The goal of the model is to classify the emotion of each song based on the BERT-generated embeddings of its lyrics.

### 5. Model Evaluation

The model is evaluated on a test set, and the accuracy of the predictions is measured. The architecture consists of:

- **LSTM layer**: For learning sequences of lyrics.
- **Fully connected layer**: For mapping hidden states to emotion labels.
- **Softmax activation**: To output probability distributions over the emotion classes.

---

## Results

Currently, the model has been evaluated but shows limited success with a test accuracy of around **2%**. This could be due to various factors, such as:

- Issues with the dataset (e.g., missing or mislabeled data).
- Misalignment between the BERT embeddings and the LSTM architecture.

Improving the dataset quality and tweaking the model parameters may lead to better results.

---

## Folder Structure

The project directory has the following structure:

```
├── data_results
│   ├── data_cleaned.csv          # Cleaned lyrics and emotion labels
│   ├── data_combined.csv         # Combined dataset with audio and lyrics features
│   └── data_with_embeddings.csv  # Dataset with BERT embeddings
├── Dockerfile                    # Docker configuration file
├── .dockerignore                 # Files to ignore during Docker build
├── main.py                       # Main script for data processing and model training
├── README.md                     # Documentation for the project
└── requirements.txt              # List of Python dependencies
```

---

## How to Run the Project

### 1. Using Docker (Recommended)

- **Build the Docker image**:

   ```bash
   docker build -t music-emotion-classification .
   ```

- **Run the Docker container**:

   ```bash
   docker run --rm -v $(pwd):/app music-emotion-classification
   ```

This will handle everything, including the installation of dependencies and execution of the project script.

### 2. Without Docker (Local Setup)

- **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

- **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

- **Run the script**:

   ```bash
   python main.py
   ```

---

## Docker Support

A Docker setup is available for this project, which includes all the necessary dependencies and configurations for running the project in a containerized environment. To use Docker:

1. **Build the Docker image**:

   ```bash
   docker build -t music-emotion-classification .
   ```

2. **Run the Docker container**:

   ```bash
   docker run --rm -v $(pwd):/app music-emotion-classification
   ```

---

## Notes

- **Data quality** is crucial for achieving good model performance. Ensure that the dataset is complete and properly labeled.
- **Model tuning**: The current architecture (BERT + LSTM) might require additional tuning and more training epochs to improve performance.
- Alternative models such as **Word2Vec**, **FastText**, or **TF-IDF** could be explored for generating embeddings instead of using BERT.

