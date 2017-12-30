############### Librairies ###############
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def get_stop_words():
    return [
        'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am',
        'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been',
        'by', 'did', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had',
        'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i',
        'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'may',
        'me', 'might', 'my', 'of', 'off', 'on', 'or', 'other', 'our', 'own',
        'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'than',
        'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this',
        'tis', 'to', 'was', 'us', 'was', 'we', 'were', 'what', 'when', 'where',
        'while', 'who', 'whom', 'why', 'will', 'would', 'yet', 'you', 'your'
    ]


def remove_words(dataset, stop_words):
    formatted_dataset = []
    for tweet in dataset:
        reduced_tweet = []
        for word in tweet.split(' '):
            if word not in stop_words:
                reduced_tweet.append(word)
        formatted_dataset.append(' '.join(reduced_tweet))
    return formatted_dataset


############### Part of Speech Tagging ###############
def part_of_speech_tagging(dataset):
    tagged_tweets = []
    for tweet in dataset:
        tokens = nltk.word_tokenize(tweet)
        tagged_tweets.append(nltk.pos_tag(tokens))
    return tagged_tweets

############### Détermination du sentiments des mots de la phrase pour obtenir le sentiment général du tweet ###############


def get_sentiment(tagged, df):
    i = 0
    dim = len(tagged)
    result_sentiment = []
    while i < dim:
        pos = neg = obj = count = 0
        for word, tag in tagged[i]:
            try:
                ss_set = None
                if 'NN' in tag and swn.senti_synsets(word):
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'VB' in tag and swn.senti_synsets(word):
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'JJ' in tag and swn.senti_synsets(word):
                    ss_set = list(swn.senti_synsets(word))[0]
                elif 'RB' in tag and swn.senti_synsets(word):
                    ss_set = list(swn.senti_synsets(word))[0]

                if ss_set:
                    pos = pos + ss_set.pos_score()
                    neg = neg + ss_set.neg_score()
                    obj = obj + ss_set.obj_score()
                    count += 1

            except:
                pass

        final_score = pos - neg
        df['final_score'] = final_score

        if final_score < 0:
            result_sentiment.append('neg')
        elif final_score > 0:
            result_sentiment.append('pos')
        elif final_score == 0:
            result_sentiment.append('other')
        i += 1

    print("\nCalcul de l'erreur \n")
    erreur = 0
    label_true = np.array(df['Sentiment'])
    dim = len(label_true)
    for j in range(dim):
        if label_true[j] != result_sentiment[j]:
            erreur = erreur + 1
        j = j + 1

    # Calcul pourcentage d'erreur :
    taux_erreur = 1 - (dim - erreur) / dim
    return taux_erreur


def main():
    dataset = dataset_manager.load('./dataset/test.csv', ',')
    dataset['Tweet'] = remove_words(dataset['Tweet'], get_stop_words())
    tagged = part_of_speech_tagging(dataset['Tweet'])
    taux_erreur = get_sentiment(tagged, dataset)

    print("Taux d'erreur : %s  " % (taux_erreur))

    return 0


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '')
    from utils import dataset_manager, metrics
    exit(main())


# changer de dictionnaire + pondéré  + methode hybride : classifier add
