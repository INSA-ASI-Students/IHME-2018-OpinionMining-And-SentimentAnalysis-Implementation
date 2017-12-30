############### Librairies ###############
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk import pos_tag


############### Dataset ###############

def loadPandaFrame(filename):
    df = pd.read_csv(filename)
    return df

############### MINUSCULE ###############


def lowerCase(df):
    train_tweet = df['Tweet']
    train_tweet_pretraitement = [item.lower() for item in train_tweet]
    df_tweet_lower = pd.DataFrame(train_tweet_pretraitement)
    df['tweet_lower'] = df_tweet_lower


############### Token extraction ###############
# def tokenization(df):
#     token = df.apply(lambda row: nltk.word_tokenize(row['tweet_lower']), axis=1)
#     df['tokenized_words'] = token
#     return df['tokenized_words'], token
def tokenization(dataset):
    tokens = []
    for tweet in dataset:
        tokens.append(nltk.word_tokenize(tweet))
    return tokens


############### Elimination des mots neutres ###############
# def removeWord(word, stop_words, df, token):
#     token_reduce = token.apply(lambda x: [word for word in x if word not in stop_words])
#     df['token_reduce'] = token_reduce


def removeWord(dataset, stop_words):
    formatted_dataset = []
    for tweet in dataset:
        reduced_tweet = []
        for word in tweet.split(' '):
            if word not in stop_words:
                reduced_tweet.append(word)
        formatted_dataset.append(' '.join(reduced_tweet))
    return formatted_dataset


############### Part of Speech Tagging ###############
# def partOfSpeechTagging(df):
#     df['tagged_tweet'] = df.apply(lambda df: nltk.pos_tag(df['token_reduce']), axis=1)
#     # Transformation en array
#     return np.array(df['tagged_tweet'])
def partOfSpeechTagging(tokens):
    tagged_tweets = []
    for row in tokens:
        tagged_tweets.append(nltk.pos_tag(row))
    return tagged_tweets

############### Détermination du sentiments des mots de la phrase pour obtenir le sentiment général du tweet ###############


def getSentiment(tagged, df):
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
        # print("Score sentiment du tweet n° %d : %s  " % (i, final_score))
        df['final_score'] = final_score

        if final_score < 0:
            result_sentiment.append('neg')
        elif final_score > 0:
            result_sentiment.append('pos')
        elif final_score == 0:
            result_sentiment.append('other')

        # print(result_sentiment[i])
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
    # print("\nDataset train.csv load \n")
    # df = loadPandaFrame("./dataset/train_ingrid.csv")
    # print("Pré-traitement \n")
    # lowerCase(df)
    dataset = dataset_manager.load('./dataset/test.csv', ',')
    # dataset = dataset_manager.format(dataset)

    # (word, token) = tokenization(dataset)
    stop_words = [
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
    dataset['Tweet'] = removeWord(dataset['Tweet'], stop_words)
    tokens = tokenization(dataset['Tweet'])
    tagged = partOfSpeechTagging(tokens)
    taux_erreur = getSentiment(tagged, dataset)

    print("Taux d'erreur : %s  " % (taux_erreur))

    return 0


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '')
    from utils import dataset_manager, metrics
    exit(main())


# changer de dictionnaire + pondéré  + methode hybride : classifier add
