from __future__ import division
import tensorflow as tf
import numpy as np
import nltk
import pickle
import random
import csv
import os
from gensim.models import FastText,Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_non_alphanum
import nltk
from nltk import sent_tokenize,word_tokenize
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
import urllib.request
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
import warnings
import flask
import json
from flask import Flask, render_template, request
warnings.filterwarnings('ignore')
from nltk import sent_tokenize,word_tokenize
from flask import jsonify
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

allWords = []
MAX_SENTENCE_LEN = 35
EMBEDDING_SIZE = 100
# number of neurons in each layer
input_num_units = MAX_SENTENCE_LEN*EMBEDDING_SIZE
hidden_num_units = 1500
output_num_units = MAX_SENTENCE_LEN
epochs = 250
batch_size = 32
learning_rate = 0.01
model_ft = FastText.load('models/bbc_ft.bin')

app = flask.Flask(__name__)


def load_model():
    global model
    model_path='models/model.json'
    json_file = open(str(model_path), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("models/model.h5")
    global graph
    graph = tf.get_default_graph()

def clean_str(string):
    string = re.sub(r"\'s", "s", string)
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
    string = re.sub(r"\n",'',string)
    string = re.sub(r"  "," ",string)
    # return string.strip().lower()
    return string.strip()

def extract_sentences_from_paragraph(paragraph):
    sentences = sent_tokenize(paragraph)
    sent_list=[]
    for s in sentences:
      if(len(s.strip()) != 0):
        s=clean_str(s)
        new_s=' '
        for w in word_tokenize(s):
          new_s.join(lemmatizer.lemmatize(w))
        sent_list.append(s)
    return sent_list

def make_array_vectorize(text):
  texts=[]
  texts.append(extract_sentences_from_paragraph(text))
  nested_list_len = lambda x: sum(len(list) for list in x)
  source_text_vectors = np.zeros((nested_list_len(texts), 3500))
  vec_idx = 0
  if(type(texts[0]) == list):
      for i in range(len(texts)):
          sentences = texts[i]
          # Get text vector
          for s in sentences:
              sentence_vector = np.array([])
            #   s=remove_stopwords(strip_punctuation(strip_non_alphanum(str(s).lower())))
              s=remove_stopwords(strip_punctuation(strip_non_alphanum(str(s))))  
              s=clean_str(s)
              for w in word_tokenize(s):
                  w=lemmatizer.lemmatize(w)
                  if(model_ft.__contains__(w)==False):
                    continue
                  if(len(sentence_vector) < MAX_SENTENCE_LEN*EMBEDDING_SIZE):
                      sentence_vector = np.append(sentence_vector, model_ft[w])
                  else:
                      break
              while(len(sentence_vector) < MAX_SENTENCE_LEN*EMBEDDING_SIZE):
                  sentence_vector = np.append(sentence_vector,np.zeros(EMBEDDING_SIZE))
              source_text_vectors[vec_idx] = sentence_vector
              vec_idx+=1
  return (source_text_vectors)

def make_text_35(text):
    words=text.split(' ')
    clean=' '
    if(len(words)>35):
        text=remove_stopwords(text)
        words=text.split(' ')
        if(len(words)>35):
            words=words[:35]
        return (clean.join(w for w in words))
    else:
        return(text)

def predict_url(url):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent,} 
    url='https://contentxtractor.com/api/v1/?a=grab&key=12jopu98&url='+str(url)
    request=urllib.request.Request(url,None,headers) #The assembled request
    response =urllib.request.urlopen(request)
    data = response.read() # The data u need
    output =json.loads(data.decode('utf8'))#json.loads(myResponse.content.decode('utf-8'))
    # print (output)
    # print(output['result'])
    # print(output['data']['text'])
    text=output['data']['text']
    # images_link=output['data']['images']
    # print(images_link)
    length_original=len(word_tokenize(text))
    print(length_original)
    arr_text=extract_sentences_from_paragraph(text)
    # print(len(arr_text))
    vec=make_array_vectorize(text)
    with graph.as_default():
        results=model.predict(vec)
    for res in results:
        word_idx=sorted(range(len(res)), key=lambda i: res[i], reverse=True)[:15]
        for i in word_idx:
            res[i]=1
    sent_sum=' '
    for res,sent in zip(results,arr_text):
        sent=str(make_text_35(sent))
        s_indv=' '.join([word for idx,word in enumerate(sent.split(" ")) if(res[idx] == 1)])
        sent_sum=str(sent_sum)+str(s_indv)+str(',')
            # print(s_indv)

    # return sent_sum
    length_summary=len(word_tokenize(sent_sum))
    print(length_summary)
    percentage_reduced=int(((length_original-length_summary)/length_original)*100)
    print(percentage_reduced)
    # return the data dictionary as a JSON response
    # images = json.dumps(images_link, sort_keys = True)
    ret_list=[sent_sum,percentage_reduced]
    return ret_list

def predict_text(text):
    # initialize the data dictionary that will be returned from the
    # view
    # data = {"success": False}
    # user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    # headers={'User-Agent':user_agent,} 
    # url='https://contentxtractor.com/api/v1/?a=grab&key=12jopu98&url='+str(url)
    # request=urllib.request.Request(url,None,headers) #The assembled request
    # response =urllib.request.urlopen(request)
    # data = response.read() # The data u need
    # output =json.loads(data.decode('utf8'))#json.loads(myResponse.content.decode('utf-8'))
    # # print (output)
    # # print(output['data']['text'])
    # text=output['data']['text']
    print(text)
    print(word_tokenize(text))
    length_original=len(word_tokenize(text))
    print(length_original)
    arr_text=extract_sentences_from_paragraph(text)
    # print(len(arr_text))
    vec=make_array_vectorize(text)
    with graph.as_default():
        results=model.predict(vec)
    for res in results:
        word_idx=sorted(range(len(res)), key=lambda i: res[i], reverse=True)[:15]
        for i in word_idx:
            res[i]=1
    sent_sum=' '
    for res,sent in zip(results,arr_text):
        sent=str(make_text_35(sent))
        s_indv=' '.join([word for idx,word in enumerate(sent.split(" ")) if(res[idx] == 1)])
        sent_sum=str(sent_sum)+str(s_indv)+str(',')
        print(sent_sum)

    # return sent_sum
    print(sent_sum)
    length_summary=len(word_tokenize(sent_sum))
    print(length_summary)
    percentage_reduced=int(((length_original-length_summary)/length_original)*100)
    print(percentage_reduced)
    # return the data dictionary as a JSON response
    ret_list=[sent_sum,percentage_reduced]
    return ret_list

@app.route('/',methods=['POST','GET'])
def home():
    # url='http://news.bbc.co.uk/2/hi/business/4236959.stm'
    # print(predict(url))
    # url = request.form['projectFilepath']
    return flask.render_template('final.html')
# @app.route('/predict', methods=['GET','POST'])

def formaturl(url):
    if not re.match('(?:http|ftp|https)://', url):
        return 'http://{}'.format(url)
    return url

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result_url = request.form['url']
        # result_text=request.form['text']
        # print(result_text)
        # if(result_url==''):
        #     predict_text(result_text)#text predict
        # if(result_text==''):
        #     summary_text=predict(result_url)#url predict
        # print(res_text)
        html = urlopen(result_url)
        bs = BeautifulSoup(html, 'html.parser')
        images = bs.find_all('img')
        data=[]
        for image in images:
            url=image['src']
            url=formaturl(url)
            data.append({'url':url})
        print(type(data))
        images_url=json.dumps(data)
        print(type(data))
        if(result_url):
            result=predict_url(result_url)
            text=result[0]
            reduced_percentage=result[1]
            return render_template("result.html",bot=text,bot1=reduced_percentage,bot2=images_url)#change this
        if(result_text):
            result=predict_text(result_text)
            text=result[0]
            reduced_percentage=result[1]
            return render_template("result.html",bot=text,bot1=reduced_percentage)#change this

@app.route('/example',methods = ['POST', 'GET'])
def example():
    url='http://news.bbc.co.uk/2/hi/business/4236959.stm'
    # print(predict(url))
    result=predict_url(url)
    text=result[0]
    reduced_percentage=result[1]
    return render_template("result.html",bot=text,bot1=reduced_percentage)#change this
    # return render_template("result.html")#change this


if __name__ == '__main__':  
    app.static_folder = 'static'
    load_model()
    app.run(debug=True)