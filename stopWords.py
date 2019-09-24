""" tokezing words"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example="this is a example about a cyberweapons project"
stop_words=set(stopwords.words("english"))
words=word_tokenize(example)

fil_sen=[]
for w in words:
    if w not in stop_words:
        fil_sen.append(w)
print(fil_sen)
