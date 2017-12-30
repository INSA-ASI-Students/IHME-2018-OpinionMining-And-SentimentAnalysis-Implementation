############### Librairies ###############
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize, word_tokenize, pos_tag


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


def swn_polarity(text, lemmatizer=WordNetLemmatizer()):
    sentiment = 0.0
    tokens_count = 0

    raw_sentences = sent_tokenize(text)

    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:
            wn_tag = bag_of_words_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)

            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)

            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

    if sentiment < 0:
        return 'neg'
    elif sentiment > 0:
        return 'pos'
    elif sentiment == 0:
        return 'other'


def predict(corpus, lemmatizer=WordNetLemmatizer()):
    predicted_sentiments = []
    for tweet in corpus:
        predicted_sentiments.append(swn_polarity(tweet, lemmatizer))
    return predicted_sentiments


def main():
    dataset = dataset_manager.load('./dataset/train.csv', ',')
    prediction = predict(dataset['Tweet'])
    taux_erreur = metrics.error_rate(dataset['Sentiment'], prediction)
    print(taux_erreur)
    return 0


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '')
    from utils import dataset_manager, metrics
    exit(main())
