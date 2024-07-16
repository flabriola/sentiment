import pandas as pd
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import sys

"""
Prepare data for training
"""

df = pd.read_csv('training.csv')

# Define X and y for splitting
x = df['text'].iloc[:100000]
y = df['sentiment'].iloc[:100000]

# Split into training and testing sets with stratification based on y (sentiment)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Tokenize text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

# Pad sequences
x_train_padded = pad_sequences(x_train_seq, padding='post', maxlen=100)
x_test_padded = pad_sequences(x_test_seq, padding='post', maxlen=100)

# Convert labels to categorical (-1, 0, 1 to 0, 1, 2)
y_train = tf.keras.utils.to_categorical(y_train + 1, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test + 1, num_classes=3)

"""
Train, test and export model
"""

# Create a neural network model
model = tf.keras.models.Sequential([
    
    # Embedding layer to find similarities between different words (context)
    tf.keras.layers.Embedding(10000, 64, input_length=100),
    
    # 4 bidirectional layers to increase hierarchical feature learning for better generalization
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout to prevent overfitting
    tf.keras.layers.Dropout(0.75),
    
    # Output layers of positive, negative and neutral
    tf.keras.layers.Dense(3, activation='softmax') 
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train_padded, y_train, epochs=10, validation_data=(x_test_padded, y_test))

# Evaluate the model
model.evaluate(x_test_padded, y_test, verbose=2)

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")