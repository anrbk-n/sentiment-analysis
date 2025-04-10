# Sentiment Analysis API with FastAPI and DistilBERT

This project implements a FastAPI-based API for sentiment analysis of text data. It leverages a fine-tuned DistilBERT model to classify the sentiment of input text as positive, negative, or neutral. The API is containerized with Docker for easy deployment and usage.

## Features

* **Sentiment Analysis:** Classifies the sentiment of text into positive, negative, or neutral categories.
* **FastAPI:** Uses FastAPI for building the API, providing high performance and automatic documentation.
* **DistilBERT:** Employs a DistilBERT model for accurate sentiment prediction.
* **Docker:** Containerized for easy deployment and consistent environment.
* **Simple Interface:** Provides a user-friendly HTML interface for testing the API.


## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/anrbk-n/sentiment-analysis.git
    cd sentiment-analysis
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the FastAPI application (locally):**

    ```bash
    python src/main.py
    ```

    * The API will be available at `http://localhost:8000`.
    * You can access the interactive documentation at `http://localhost:8000/docs`.

## Docker

### Prerequisites

* [Docker](https://www.docker.com/) must be installed on your system.

### Building the Docker image

```bash
docker build -t sentiment-analysis .
`````
### Running the Docker container
```bash
docker run -p 8000:8000 sentiment-analysis-api
`````

### The API will be available at http://localhost:8000.

## Model
* ### Model: ```DistilBertForSequenceClassification```
* ### Fine-tuned on dataset from Hugging Face: 
  `https://huggingface.co/datasets/DmitrySharonov/ru_sentiment_neg_pos_neutral`
* ### Accuracy: ``78.14%``
