import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import ISRIStemmer
# import gensim
# from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# import fasttext

###############################################################################
# read data and store it in dataframe
# train_data = pd.read_excel('./train.xlsx')

###############################################################################
def remove_special_chars(text):
    pattern = re.compile(r'[^\w\s\u0600-\u06FF]+', re.UNICODE)
    filtered_text = re.sub(pattern, '', text)
    return filtered_text

###############################################################################
def remove_num(text):
    pattern = r'\d+'
    filtered_text = re.sub(pattern, '', text)
    return filtered_text

###############################################################################
def remove_punc(text):
    # pattern = r'[^\w\s]'
    pattern = r'[^\w\s_]'
    filtered_text = re.sub(pattern, '', text)
    return filtered_text

###############################################################################
def remove_non_arabic(text):
    pattern = re.compile(r'[^\u0600-\u06FF\s]+', re.UNICODE)
    filtered_text = re.sub(pattern, '', text)
    return filtered_text

###############################################################################
def remove_underscore(input_string):
    return input_string.replace("_", "")

###############################################################################
def tokenization(text):
    return nltk.word_tokenize(text)

###############################################################################
def remove_stopwords(text):
    stop_words = set(stopwords.words('arabic'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

###############################################################################
def stemming(text):
  stemmer = ISRIStemmer()
  words = word_tokenize(text)
  stemmed_words = [stemmer.stem(word) for word in words]
  return ' '.join(stemmed_words)

###############################################################################
# def word_embedding(text):
#     model_path = 'cc.ar.300.bin' 
#     model = fasttext.load_model(model_path)
#     word_embeddings = [model.get_word_vector(word) for word in text]
#     return ' '.join(word_embeddings)
###############################################################################
# train_data['cleaned_text'] = train_data['review_description'].apply(remove_special_chars)
# train_data['cleaned_text2'] = train_data['cleaned_text'].apply(remove_num)
# train_data['cleaned_text3'] = train_data['cleaned_text2'].apply(remove_punc)
# train_data['cleaned_text4'] = train_data['cleaned_text3'].apply(remove_non_arabic)
# train_data['cleaned_text5'] = train_data['cleaned_text4'].apply(remove_underscore)
# train_data['cleaned_text6'] = train_data['cleaned_text5'].apply(remove_stopwords)
# train_data['cleaned_text7'] = train_data['cleaned_text6'].apply(stemming)
# train_data['cleaned_text8'] = train_data['cleaned_text7'].apply(tokenization)

#Functions To Preprocess Dataset
def clean_reviews(text):
    #remove_special_chars
    pattern = re.compile(r'[^\w\s\u0600-\u06FF]+', re.UNICODE)
    text = re.sub(pattern, '', text)
    
    #remove_num
    text = re.sub(r'\d+', '', text)

    #remove_punc
    text = re.sub(r'[^\w\s_]', '', text)
    
    #remove_non_arabic
    pattern = re.compile(r'[^\u0600-\u06FF\s]+', re.UNICODE)
    text = re.sub(pattern, '', text)

    #remove_repeating_char
    text= re.sub(r'(.)\1+', r'\1', text)

    
    #remove_underscore
    text=text.replace("_", "")
    #remove_stopwords
    stop_words = set(stopwords.words('arabic'))
    words = word_tokenize(text)
    text = [word for word in words if word.lower() not in stop_words]
    text_after_remove_stop_words=' '.join(text)
    
    #stemming
    stemmer = ISRIStemmer()
    words = word_tokenize(text_after_remove_stop_words)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)
     

# read train_dataset
train_data = pd.read_excel('./train.xlsx')
# clean the data
reviews = train_data['review_description'].apply(clean_reviews)


# ratings = train_dataset['rating']
max_fatures = 10 # our model will remeber last 100 words
tokenizer = Tokenizer (num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(reviews)
pad_train = tokenizer.texts_to_sequences (reviews)
pad_train= pad_sequences (pad_train)

###############################################################################
# Define a custom dataset class
# class CustomDataset(data.Dataset):
#     def __init__(self, src_data, tgt_data):
#         self.src_data = src_data
#         self.tgt_data = tgt_data
#         print('Shape of src_data:', src_data.shape)
#         print('Shape of tgt_data:', tgt_data.shape)


#     def __len__(self):
#         return len(self.src_data)

#     def __getitem__(self, index):
#         # Filter out examples with empty target sequences
#         while not self.tgt_data.iloc[index]:
#             index = (index + 1) % len(self)
#         return self.src_data.iloc[index], self.tgt_data.iloc[index]
################################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
   
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
##################################################################################
src_vocab_size = 100
tgt_vocab_size = 100
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 43
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
import tensorflow as tf

# src_data =  tf.convert_to_tensor(pad_train) # tensor = tf.convert_to_tensor(array)
# src_data = torch.from_numpy(pad_train)

src_data = tf.convert_to_tensor(pad_train)

# Expand the dimensions of the tensor
src_data = tf.expand_dims(src_data, axis=0)

tgt_data =train_data['rating']

tgt_data =  tf.convert_to_tensor(tgt_data)
# src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()


for epoch in range(10):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data)
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data.contiguous().view(-1))
    loss.backward()
    optimizer.step()
    # Calculate accuracy
    _, predicted = output.max(2)
    correct = (predicted == tgt_data).sum().item()
    total = (tgt_data != 0).sum().item()  # Exclude padding tokens from the total count
    accuracy = correct / total
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}")
    # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")   


# for epoch in range(10):
#     optimizer.zero_grad()
#     output = transformer(src_data, tgt_data[:,:-1])
#     loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     # Calculate accuracy
#     _, predicted = output.max(2)
#     correct = (predicted == tgt_data[1:]).sum().item()
#     total = (tgt_data[1:] != 0).sum().item()  # Exclude padding tokens from the total count
#     accuracy = correct / total
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}")
#     # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")   

