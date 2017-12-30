############### Librairies ###############
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize, word_tokenize, pos_tag


############### Dataset ###############

def loadPandaFrame(filename):
    df = pd.read_csv(filename)
    return df['Tweet'], df['Sentiment']


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


def convertSentimentIntoClass(dataset):
    converted_dataset = []
    for i in range(len(dataset)):
        if dataset[i] == 'pos':
            converted_dataset.append(1)
        elif dataset[i] == 'neg':
            converted_dataset.append(0)
        elif dataset[i] == 'other':
            converted_dataset.append(2)
    return converted_dataset


def swn_polarity(text, lemmatizer=WordNetLemmatizer()):
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


# def start(dataset):


def main():
    print("\nDataset train.csv load \n")
    (tweets, labels) = loadPandaFrame("./StanceDataset/train_ingrid.csv")
    lemmatizer = WordNetLemmatizer()

    labels = convertSentimentIntoClass(labels)
    dim = len(tweets)
    #print (swn_polarity(train_X[1],lemmatizer), train_y[1])
    nb_errors = 0
    for j in range(dim):
        if swn_polarity(tweets[j], lemmatizer) != labels[j]:
            nb_errors = nb_errors + 1
        j = j + 1
    taux_erreur = 1 - (dim - nb_errors) / dim
    print(taux_erreur)
    return 0


if __name__ == '__main__':
    exit(main())
