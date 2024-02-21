#import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import ISRIStemmer
from sklearn import preprocessing 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer,Dense, Bidirectional, LSTM, Dropout, Activation, Embedding
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 


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
     

#read the train dataset
train_dataset = pd.read_excel('train.xlsx') 

#clean the data
reviews = train_dataset['review_description'].apply(clean_reviews)
ratings = train_dataset['rating']

max_fatures = 100 #our model will remeber last 100 words
tokenizer = Tokenizer (num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(reviews)
pad_train = tokenizer.texts_to_sequences (reviews)
pad_train= pad_sequences (pad_train) #padding to make all sentence at same length

#encode ratings 
train_rating = ratings + 1 # Add 1 to convert -1->0 , 1->2 , 0->1 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pad_train, train_rating, test_size=0.3,stratify=train_rating)

embed_dim =4
max_fatures = 100
model1 = Sequential()
model1.add(Embedding(max_fatures, embed_dim, input_length = len(pad_train[0]),trainable=True))
model1.add(LSTM(10,trainable=True))
model1.add(Dense(3, activation='softmax'))

#compile the model
opt = keras.optimizers.Adam(learning_rate=0.01)
model1.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model1.fit(X_train,y_train, epochs=10, batch_size=32)
loss,accuracy = model1.evaluate(X_test,y_test)
print('model1 loss = ',loss)
print('model1 accurcy = ',accuracy)

#read test_dataset
test_dataset = pd.read_csv('test _no_label.csv') 
# clean the data
cleaned_reviews = test_dataset['review_description'].apply(clean_reviews)
pad_test = tokenizer.texts_to_sequences (cleaned_reviews)
pad_test= pad_sequences (pad_test, maxlen=len(pad_train[0]))  # padding to make all sentence at same length
predicted_ratings_model1=model1.predict(pad_test)

import numpy as np
y_new_pred_original=[]
y_new_pred_original = np.argmax(predicted_ratings_model1, axis=1)
y_new_pred_original=y_new_pred_original-1
#print(y_new_pred_original)

#unique_values = list(set(y_new_pred_original))
#print('num unique values = ',unique_values)
#print('num unique values = ',len(unique_values))

# Create a new DataFrame with the cleaned reviews and predicted ratings
submtion_csv = pd.DataFrame({'ID':  range(1, 1001),
                          'Predicted_Ratings': y_new_pred_original})

# Save the DataFrame to a new CSV file
submtion_csv.to_csv('lstm.csv', index=False)  # Update with the desired filename and path