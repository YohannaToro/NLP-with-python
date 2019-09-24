import nltk
import random
from nltk.corpus import movie_reviews

documents=[(list(movie_reviews.words(fileid)),category)
           for category in movie_rewards.categories()
           for fileid in movie_rewards.fileids(category)
           ]
random.shuffle(documents)

all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words=nltk.FreqDist(all_words)
#print(all_words.most_common(15))
print(all_words["stupid"])

