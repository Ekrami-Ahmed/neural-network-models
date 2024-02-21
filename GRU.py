# Import libraries
import pandas as pd
import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import ISRIStemmer
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

##############################################################################################################################################################
# Data preprocessing
def clean_reviews(text):
    # Remove the special chars
    pattern = re.compile(r'[^\w\s\u0600-\u06FF]+', re.UNICODE)
    text = re.sub(pattern, '', text)

    # Remove nums
    text = re.sub(r'\d+', '', text)

    # Remove punc
    text = re.sub(r'[^\w\s_]', '', text)

    # Remove non-Arabic
    pattern = re.compile(r'[^\u0600-\u06FF\s]+', re.UNICODE)
    text = re.sub(pattern, '', text)

    # Remove repeating chars
    text = re.sub(r'(.)\1+', r'\1', text)

    # Remove underscore
    text = text.replace("_", "")

    # Remove stopwords
    stop_words = set(stopwords.words('arabic'))
    words = word_tokenize(text)
    text = [word for word in words if word.lower() not in stop_words]
    text_after_remove_stop_words = ' '.join(text)

    # Stemming
    stemmer = ISRIStemmer()
    words = word_tokenize(text_after_remove_stop_words)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

##############################################################################################################################################################
# Read the train dataset
train_dataset = pd.read_excel('train.xlsx')

##############################################################################################################################################################
# Clean the data
reviews = train_dataset['review_description'].apply(clean_reviews)
ratings = train_dataset['rating']

max_features = 100  # Our model will remember the last 100 words
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(reviews)
pad_train = tokenizer.texts_to_sequences(reviews)
pad_train = pad_sequences(pad_train)  # Padding to make all sentences the same length

##############################################################################################################################################################
# Encode ratings
train_rating = ratings + 1  # Add 1 to convert -1->0, 1->2, 0->1

##############################################################################################################################################################
X_train, X_test, y_train, y_test = train_test_split(pad_train, train_rating, test_size=0.3, stratify=train_rating)

embed_dim = 4
model_gru = Sequential()
model_gru.add(Embedding(max_features, embed_dim, input_length=len(pad_train[0]), trainable=True))
model_gru.add(GRU(128, activation='relu'))
model_gru.add(Dense(3, activation='softmax'))

##############################################################################################################################################################
# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.01)
model_gru.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

##############################################################################################################################################################
# Train the model
model_gru.fit(X_train, y_train, epochs=10, batch_size=32)

##############################################################################################################################################################
# Evaluate the model
loss_gru, accuracy_gru = model_gru.evaluate(X_test, y_test)
print('Model GRU loss = ', loss_gru)
print('Model GRU accuracy = ', accuracy_gru)

##############################################################################################################################################################
# Read the test dataset
test_dataset = pd.read_csv('test _no_label.csv')

##############################################################################################################################################################
# Clean the data
cleaned_reviews = test_dataset['review_description'].apply(clean_reviews)
pad_test = tokenizer.texts_to_sequences(cleaned_reviews)
pad_test = pad_sequences(pad_test, maxlen=len(pad_train[0]))  # Padding to make all sentences the same length

##############################################################################################################################################################
# Predict ratings using GRU
predicted_ratings_gru = model_gru.predict(pad_test)

y_new_pred_gru = np.argmax(predicted_ratings_gru, axis=1)
y_new_pred_gru = y_new_pred_gru - 1

##############################################################################################################################################################
# Create a new data frame with the cleaned reviews and predicted ratings for GRU
submission_csv_gru = pd.DataFrame({'ID': range(1, 1001),
                                   'Predicted_Ratings': y_new_pred_gru})

# Save the data frame to a CSV file
submission_csv_gru.to_csv('gru.csv', index=False)  # Update with the desired filename and path
