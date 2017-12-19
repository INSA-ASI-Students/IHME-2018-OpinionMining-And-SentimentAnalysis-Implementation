############### Librairies ###############
import math as m
import numpy as np
import random
import os
import nltk
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize, word_tokenize, pos_tag


############### Dataset ###############

def loadPandaFrame(filename):
    df=pd.read_csv(filename)
    test_tweet = df['Tweet']
    train_tweet = df['Tweet']
    train_sentiment = df['Sentiment']
    test_sentiment = df['Sentiment']
    return train_tweet,train_sentiment,test_tweet,test_sentiment

def bag_of_words_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def convertSentimentIntoClass(train_y,test_y):
    for i in range(len(train_y)):
        if train_y[i] == 'pos':
            train_y[i] = 1
        elif  train_y[i] == 'neg':
            train_y[i] = 0
        elif  train_y[i] == 'other':
            train_y[i] = 2

    for i in range(len(test_y)):
        if test_y[i] == 'pos':
            test_y[i] = 1
        elif  test_y[i] == 'neg':
            test_y[i] = 0
        elif  test_y[i] == 'other':
            test_y[i] = 2



def swn_polarity(text,lemmatizer):
    sentiment = 0.0
    tokens_count = 0

    raw_sentences = sent_tokenize(text)
    #print ("raw_sentences")
    #print (raw_sentences)

    for raw_sentence in raw_sentences:
        #print ("word_tokenize(raw_sentence)")
        #print (word_tokenize(raw_sentence))
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        #print ("tagged_sentence")
        #print (tagged_sentence)

        for word, tag in tagged_sentence:
            wn_tag = bag_of_words_wn(tag)
            #print ("wn_tag")
            #print (wn_tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            #print ("lemma")
            #print (lemma)

            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)


            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            #print ("synset")
            #print (synset)
            swn_synset = swn.senti_synset(synset.name())
            #print ("swn_synset")
            #print (swn_synset)

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
            #print ("sentiment")
            #print (sentiment)

    if sentiment < 0:
        return 0
    elif sentiment > 0:
        return 1
    elif sentiment == 0:
        return 2




def main():
    print("\nDataset train.csv load \n")
    (train_X,train_y,test_X,test_y) = loadPandaFrame("../StanceDataset/train_ingrid.csv")
    lemmatizer = WordNetLemmatizer()

    convertSentimentIntoClass(train_y,test_y)
    dim = len(train_X)
    #print (swn_polarity(train_X[1],lemmatizer), train_y[1])
    Nberreur = 0
    for j in range(dim):
        if swn_polarity(train_X[j],lemmatizer) != train_y[j]:
            Nberreur=Nberreur+1
        j=j+1
    taux_erreur = 1-(dim-Nberreur)/dim
    print(taux_erreur)
    return 0


if __name__ == '__main__':
    exit(main())
