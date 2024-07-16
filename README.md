# Sentiment Analysis Project

This project was developed as a practice to test my skills with interacting with machine learning models, data manipulation/engineering, and Python programming. It is part of a project for Harvard's CS50P online course. The project is very simple and intended for practice purposes.

## Overview

The goal of this project is to perform sentiment analysis on textual data. Initially, the project was meant to get sentiment from topics on the platform X (formerly Twitter). However, since the API is paid, I switched to news articles instead. Although X is better for extracting sentiment on topics due to its opinion-based nature, this project uses news articles for the same purpose.

## Project Structure

- `data.py`: Handles data preparation and parsing of all datasets to include only a body of text and sentiment score (-1, 0, 1).
- `model.h5` & `model2.h5`: Pre-trained models for sentiment analysis.
- `project.py`: The main script that fetches articles based on user input, analyzes their sentiment, and prints the results.
- `rating.csv`, `Twitter_Data.csv`, `train.jsonl`: Datasets used for training the sentiment analysis model.
- `requirements.txt`: Lists all the dependencies required for the project.
- `sentiment.py`: Prepares data for training, trains the model, and saves it.
- `test_project.py`: Contains unit tests for the functions in `project.py`.

## Datasets

The datasets used in this project were found on Kaggle. To use the full program, you will need to download them from the following links:

- `Twitter_Data.csv`: [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset)
- `rating.csv`: [Global News Dataset](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset?select=rating.csv)
- `train.jsonl`: [News Articles Sentiment](https://www.kaggle.com/datasets/fhamborg/news-articles-sentiment?select=train.jsonl)

## Usage

1. **Data Preparation**: The `data.py` script reads and processes the datasets (`Twitter_Data.csv`, `rating.csv`, `train.jsonl`) and combines them into a single CSV file (`training.csv`) for training the model.

2. **Training the Model**: The `sentiment.py` script trains a neural network model on the prepared data and saves the trained model to a file (`model.h5`).

3. **Running the Sentiment Analysis**: The `project.py` script is the main entry point. It fetches news articles based on user input, tokenizes and pads the text, and uses the trained model to predict the sentiment of each article.

4. **Testing**: The `test_project.py` script contains unit tests to ensure the functionality of the main functions in `project.py`.

## Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
