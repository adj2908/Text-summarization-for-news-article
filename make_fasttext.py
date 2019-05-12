import re
import pandas as pd
from textblob import Word
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import FastText,Word2Vec
# nltk.downloaf('pu')


def clean_str(string):
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

data = pd.read_csv('bbc_news.csv',header=0,encoding = 'unicode_escape')
x = data['news'].tolist()
for index,value in enumerate(x):
    print("processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])
    # break

# text = []
# for i in x:
#     doc = str(i).lower()
#     tokens = [token.text for token in doc]
#     text.append(tokens)

# print(len(text))
ct=0
text_vocab=[]
# for i,ctext in enumerate(df1["ctext"]):
for text in x:
    text=str(text).lower()
    articles_tokens=[]
    for token in word_tokenize(text):
        articles_tokens.append(token)
    ct+=1
    print(ct)
    text_vocab.append(articles_tokens)
print(len(x))
print(len(text_vocab))

model_ted = FastText(text_vocab, size=100, window=5, min_count=5, workers=4)

print(model_ted.most_similar('china'))
print(model_ted.most_similar('bjp'))
print(model_ted.most_similar('timeswarner'))
#model_ted.save('bbc_ft.bin')
print(model_ted['china'])
print(model_ted['bjp'])
# print(model_ted[''])
# articles_tokens=[]
# for text in x:
#     articles_tokens.append([word for word in word_tokenize(str(x).lower().replace("."," "))])
#     print('')
# print(x[0])
# print(articles_tokens[0 ])