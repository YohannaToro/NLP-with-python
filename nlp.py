import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet
from nltk.corpus import movie_reviews

example="This is an example showing of cyberweapons"
stop_words=set(stopwords.words("english"))

def filter_sentence(text):
    text=text.lower()
    words=word_tokenize(text)
    filtered_sentence=[w for w in words if not w in stop_words]
    return filtered_sentence

#Deruivados de palabras
def stemmig_text(text):
    ps=PorterStemmer()
    words=word_tokenize(text)
    filtered_sentence=[]
    for w in words:
        filtered_sentence.append(ps.stem(w))
    return filtered_sentence
#chunking train model

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custom_sent_tokenizer.tokenize(sample_text)
def process_content():
    try:
        for i in tokenized:
            #print(i)
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            """chunked words"""
            #chunkGram=r"""Chunk:{<RB.?>*<VB.?>*<NNP><NN>?}"""
            #chunkParser= nltk.RegexpParser(chunkGram)

            #chunked=chunkParser.parse(tagged)
            #print(chunked)
            """ Named Entity Recognition """
            namedEnt=nltk.ne_chunk(tagged,binary=True)
            namedEnt.draw()
    except Exception as e:
        print("hello mk")
        print(str(e))

"""Lemmatizing  similar to stemming"""
lemmatizer=WordNetLemmatizer()
#print(lemmatizer.lemmatize("better",pos="a"))
#cambia el sentido de palabras better --> good

"""WordNet"""
def definition_words(word):
    syns=wordnet.synsets(word)
    
    #definition
    #print(syns[0].definition())
    #examples
    #print(syns[0].examples())
def synonyms_antonyms(word):
    syns=wordnet.synsets("program")
    synonyms=[];antonyms=[]
    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            synonyms.append(i.name())
            if i.antonyms():
                antonyms.append(i.antonyms()[0].name())
    print(set(synonyms))
def semmantic_similar(w1,w2):
    word1=wordnet.synset(w1+".n.01")
    word2=wordnet.synset(w2+".n.01")
    return word1.wup_similarity(word2)

"""text classification"""

def text_classification():
    documents=[(list(movie_reviews.words(fileid)),category)
               for category in movie_reviews.categories()
               for fileid in movie_reviews.fileids(category)
               ]
    random.shuffle(documents)
    print(documents[1])
#text prepare
def frequence_words():
    all_words=[]
    for w in movie_reviews.words():
        all_words.append(w.lower())
    all_words=nltk.FreqDist(all_words)
    return all_words.most_common(15)
def frequence_specific_word(text):
    all_words=[]
    for w in movie_reviews.words():
        all_words.append(w.lower())
    all_words=nltk.FreqDist(all_words)
    return all_words[text]
