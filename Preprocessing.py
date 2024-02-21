import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import ISRIStemmer

###############################################################################
# read data and store it in dataframe
train_data = pd.read_excel('train.xlsx')

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
train_data['cleaned_text'] = train_data['review_description'].apply(remove_special_chars)
train_data['cleaned_text2'] = train_data['cleaned_text'].apply(remove_num)
train_data['cleaned_text3'] = train_data['cleaned_text2'].apply(remove_punc)
train_data['cleaned_text4'] = train_data['cleaned_text3'].apply(remove_non_arabic)
train_data['cleaned_text5'] = train_data['cleaned_text4'].apply(remove_underscore)
train_data['cleaned_text6'] = train_data['cleaned_text5'].apply(remove_stopwords)
train_data['cleaned_text7'] = train_data['cleaned_text6'].apply(stemming)
train_data['cleaned_text8'] = train_data['cleaned_text7'].apply(tokenization)

###############################################################################
# # Example usage
# emoji_text="😘😊♡💩❤👌🤍😁🤌😘♥😂👀⭐🤞💜😒😛💞😚😊😘💪💓❤️💯👌👍😹⁩😻😉💕😜😊💙💔😃😆🥰😍😋😱😣😕💓😤🌹😎🖤💗"
# pun_text="!@#$%^&*()_+-=*/|\'"":;÷×{}[].?<>~"
# number_text="1234567890٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩"
# mix_text = "_❣😘‼️😊♡💩❤👌🤍{}[]😁 –🤌😘♥😂!👀@⭐#🤞$💜%😒^😛&💞*😚)😊(😘-💪_💓+⁦❤️-💯/👌\👍|😹÷⁩😻×😉:💕;😜""😊''💙>💔<😃.😆,🥰?😍~😋😱😣😕💓😤🌹😎🖤💗 :-)"

# ###############################################################################
# cleaned_text = remove_underscore(mix_text)







