############### Librairies ###############
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, pos_tag_sents


############### Dataset ###############

df=pd.read_csv("../StanceDataset/train_ingrid.csv")
print("\nDataset train.csv load \n")


# Passage des tweets en minuscule
print("Pré-traitement \n")
train_tweet = df['Tweet']
train_tweet_pretraitement = [item.lower() for item in train_tweet]
df_tweet_lower = pd.DataFrame(train_tweet_pretraitement)
df['tweet_lower']=df_tweet_lower


############### Token extraction ###############
print("Tokenization \n")
token = df.apply(lambda row: nltk.word_tokenize(row['tweet_lower']), axis=1)
df['tokenized_words']=token
words=df['tokenized_words']

############### Elimination des mots neutres ###############
stop_words=['a', 'able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','by','did','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','may','me','might','my','of','off','on','or','other','our','own','rather','said','say','says','she','should','since','so','than','that','the','their','them','then','there','these','they','this','tis','to','was','us','was','we','were','what','when','where','while','who','whom','why','will','would','yet','you','your','They','Look','Good','A', 'Able','About','Across','After','All','Almost','Also','Am','Among','An','And','Any','Are','As','At','Be','Because','Been','By','Did','Else','Ever','Every','For','From','Get','Got','Had','Has','Have','He','Her','Hers','Him','His','How','However','I','If','In','Into','Is','It','Its','Just','Least','Let','May','Me','Might','My','Of','Off','On','Or','Other','Our','Own','Rather','Said','Say','Says','She','Should','Since','So','Than','That','The','Their','Them','Then','There','These','They','This','Tis','To','Was','Us','Was','We','Were','What','When','Where','While','Who','Whom','Why','Will','Would','Yet','You','Your','!','@','#','"','$','(','.',')']
token_reduce = token.apply(lambda x: [word for word in x if word not in stop_words])
df['token_reduce']=token_reduce
#print(df['token_reduce'])

############### Part of Speech Tagging ###############
print("Part of Speech Tagging \n")
df['tagged_tweet'] = df.apply(lambda df:nltk.pos_tag(df['token_reduce']),axis=1)
#print(df['tagged_tweet'])

#Transformation en array
tagged=np.array(df['tagged_tweet'])


############### Détermination du sentiments des mots de la phrase pour obtenir le sentiment général du tweet ###############
# Init
print('Détermination du sentiment de chaque token puis du sentiment général \n')
i=0
dim = len(tagged)
result_sentiment = []


while i < dim:
    pos=neg=obj=count=0
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
                pos=pos+ss_set.pos_score()
                neg=neg+ss_set.neg_score()
                obj=obj+ss_set.obj_score()
                count+=1

        except:
            pass

    final_score=pos-neg
    print("Score sentiment du tweet n° %d : %s  " % (i, final_score))
    df['final_score']=final_score

    if final_score < 0 :
        result_sentiment.append('neg')
    elif final_score > 0 :
        result_sentiment.append('pos')
    elif final_score == 0 :
        result_sentiment.append('other')


    print(result_sentiment[i])
    i+=1

############### Comparaison du labels déterminé et du vraie label ###############

print("\nCalcul de l'erreur \n")
erreur = 0

label_true=np.array( df['Sentiment'])

dim = len(label_true)

for j in range(dim):
    if label_true[j] != result_sentiment[j]:
        erreur=erreur+1
    j=j+1

# Calcul pourcentage d'erreur :
print(dim)
print(erreur)

taux_erreur = (dim-erreur)/dim

print("Taux d'erreur : %s  " % (taux_erreur))
