#import libraries
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
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

##############################################################################################################################################################
#data preprocessing
def clean_reviews(text):
    #remove the special chars
    pattern = re.compile(r'[^\w\s\u0600-\u06FF]+', re.UNICODE)
    text = re.sub(pattern, '', text)
    
    #remove nums
    text = re.sub(r'\d+', '', text)

    #remove punc
    text = re.sub(r'[^\w\s_]', '', text)
    
    #remove non arabic 
    pattern = re.compile(r'[^\u0600-\u06FF\s]+', re.UNICODE)
    text = re.sub(pattern, '', text)

    #remove repeating chars
    text= re.sub(r'(.)\1+', r'\1', text)

    #remove underscore
    text=text.replace("_", "")

    #remove stopwords
    stop_words = set(stopwords.words('arabic'))
    words = word_tokenize(text)
    text = [word for word in words if word.lower() not in stop_words]
    text_after_remove_stop_words=' '.join(text)
    
    #stemming
    stemmer = ISRIStemmer()
    words = word_tokenize(text_after_remove_stop_words)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

##############################################################################################################################################################   
#read the train dataset
train_dataset = pd.read_excel('train.xlsx')

##############################################################################################################################################################
#clean the data
reviews = train_dataset['review_description'].apply(clean_reviews)
ratings = train_dataset['rating']

max_features = 100  #our model will remember the last 100 words
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(reviews)
pad_train = tokenizer.texts_to_sequences(reviews)
pad_train = pad_sequences(pad_train)  #padding to make all sentences the same length

##############################################################################################################################################################
#encode ratings
train_rating = ratings + 1  # Add 1 to convert -1->0, 1->2, 0->1

##############################################################################################################################################################
X_train, X_test, y_train, y_test = train_test_split(pad_train, train_rating, test_size=0.3, stratify=train_rating)

embed_dim = 4
model_cnn = Sequential()
model_cnn.add(Embedding(max_features, embed_dim, input_length=len(pad_train[0]), trainable=True))
model_cnn.add(Conv1D(128, 5, activation='relu'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(3, activation='softmax'))

##############################################################################################################################################################
#compile the model
opt = keras.optimizers.Adam(learning_rate=0.01)
model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

##############################################################################################################################################################
#train the model
model_cnn.fit(X_train, y_train, epochs=10, batch_size=32)

##############################################################################################################################################################
#evaluate the model
loss, accuracy = model_cnn.evaluate(X_test, y_test)
print('Model CNN loss = ', loss)
print('Model CNN accuracy = ', accuracy)

##############################################################################################################################################################
#read the test dataset
test_dataset = pd.read_csv('test _no_label.csv')

##############################################################################################################################################################
#clean the data
cleaned_reviews = test_dataset['review_description'].apply(clean_reviews)
pad_test = tokenizer.texts_to_sequences(cleaned_reviews)
pad_test = pad_sequences(pad_test, maxlen=len(pad_train[0]))  #padding to make all sentences the same length

##############################################################################################################################################################
#predict ratings using CNN 
predicted_ratings_cnn = model_cnn.predict(pad_test)

y_new_pred_cnn = np.argmax(predicted_ratings_cnn, axis=1)
y_new_pred_cnn = y_new_pred_cnn - 1

##############################################################################################################################################################
#create a new data frame with the cleaned reviews and predicted ratings
submission_csv_cnn = pd.DataFrame({'ID': range(1, 1001),
                                   'Predicted_Ratings': y_new_pred_cnn})

#save the data frame to a CSV file
submission_csv_cnn.to_csv('cnn.csv', index=False)  # Update with the desired filename and path
