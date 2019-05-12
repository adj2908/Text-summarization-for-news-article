import numpy as np
import os
import nltk
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_non_alphanum
import pickle
import re
from nltk import sent_tokenize
# ['quarterly profits at us media giant time warner jumped to', 'for the three months to december from year earlier', 'the firm which is now one of the biggest investors in google benefited from sales of high speed internet connections and higher advert sales', 'timewarner said fourth quarter sales rose to', 'from', '', 'its profits were buoyed by one off gains which offset a profit dip at warner bros and less users for aol', 'time warner said on friday that it now owns of search engine google', 'but its own internet business aol had has mixed fortunes', 'it lost subscribers in the fourth quarter profits were lower than in the preceding three quarters', 'however the company said aol underlying profit before exceptional items rose on the back of stronger internet advertising revenues', 'it hopes to increase subscribers by offering the online service free to timewarner internet customers and will try to sign up aol existing customers for high speed broadband', 'timewarner also has to restate and results following a probe by the us securities exchange commission sec which is close to concluding', 'time warner fourth quarter profits were slightly better than analysts expectations', 'but its film division saw profits slump to helped by box office flops alexander and catwoman a sharp contrast to year earlier when the third and final film in the lord of the rings trilogy boosted results', 'for the full year timewarner posted a profit of', 'up from its performance while revenues grew', 'to', '', 'our financial performance was strong meeting or exceeding all of our full year objectives and greatly enhancing our flexibility chairman and chief executive richard parsons said', 'for timewarner is projecting operating earnings growth of around and also expects higher revenue and wider profit margins', 'timewarner is to restate its accounts as part of efforts to resolve an inquiry into aol by us market regulators', 'it has already offered to pay to settle charges in a deal that is under review by the sec', 'the company said it was unable to estimate the amount it needed to set aside for legal reserves which it previously set at', 'it intends to adjust the way it accounts for a deal with german music publisher bertelsmann purchase of a stake in aol europe which it had reported as advertising revenue', 'it will now book the sale of its stake in aol europe as a loss on the value of that stake']
# ad sales boost time warner profit
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
    string = re.sub(r"\n",'',string)
    string = re.sub(r"  ",' ',string)
    return string.strip().lower()


def extract_sentences_from_paragraph(paragraph):
    sentences = sent_tokenize(paragraph)
    return [clean_str(s) for s in sentences if len(s.strip()) != 0]

path='BBC_NEWS'
texts = []
for subdir, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".txt"):
            if(subdir[:22]==r'BBC_NEWS\News_Articles'):
                with open(os.path.join(subdir, filename), 'r') as f:
                    x = f.readlines()
                    paragraph = " ".join(x[1:]).strip()
                    texts.append(extract_sentences_from_paragraph(paragraph))
        # break
# ct=0
# texts_cl=[]
# for lst in texts:
#     lst_cl=[]
#     for x in lst:
#         x=x.replace('\n','')
#         x=x.replace('  ',' ')
#         # print(x)
#         # ct+=1
#         x=strip_punctuation(x)
#         lst_cl.append(x)
#     texts_cl.append(lst_cl)
    # break

# print(texts_cl[0])
# print(summaries[0])

# with open('texts.pkl','wb') as f:
#     pickle.dump(texts_cl,f)

# with open('summary.pkl','wb') as fo:
#     pickle.dump(summaries,fo)

# print(ct)
# print(len(texts_cl[0]))
# print(len(texts_cl))
# print('###')
# print(summaries[0])
# print(texts_cl)



path='BBC_NEWS'
# texts = []
summaries = []
for subdir, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".txt"):
            if(subdir[:19]==r'BBC_NEWS\ASummaries'):
                with open(os.path.join(subdir, filename), 'r') as f:
                    x = f.readlines()
                    s=x[0].strip()
                    s=clean_str(s)
                    summaries.append(s)

with open('texts.pkl','wb') as f:
    pickle.dump(texts,f)

with open('summary.pkl','wb') as fo:
    pickle.dump(summaries,fo)

print(len(texts[0]))
print(len(texts))
print(len(summaries))
print('###')
print(summaries[0])
print(texts[0])