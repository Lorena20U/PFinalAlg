import nltk
import pandas as pd
import numpy as np
import random
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from os import listdir
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


path_ham = "src/ham/"
listadoArchivosHam = [path_ham+f for f in listdir(path_ham) if f.endswith('.txt')]

path_spam = "src/spam/"
listadoArchivosSpam = [path_spam+f for f in listdir(path_spam) if f.endswith('.txt')]

#Lectura de la data
def abrir(texto):
    with open(texto, 'r', errors='ignore') as f2:
        data = f2.read()
        data = word_tokenize(data)
    return data

#Lista del ham
listadoHam = list(map(abrir, listadoArchivosHam))
#Lista del spam
listadoSpam = list(map(abrir, listadoArchivosSpam))



#Separacion de las palabras comunes
todasPal = nltk.FreqDist([w for tokenlist in listadoHam+listadoSpam for w in tokenlist])
princPal = todasPal.most_common(250)

#Agregan bigramas, diagramas
bigTexto = nltk.Text([w for token in listadoHam+listadoSpam for w in token])
bigramas = list(nltk.bigrams(bigTexto))
princBig = (nltk.FreqDist(bigramas)).most_common(250)


def caract(doc):
    docPalabras = set(doc)
    bigram = set(list(nltk.bigrams(nltk.Text([token for token in doc]))))
    caracts = {}
    for palab, j in princPal:
        caracts['Contiene Palabras: ({})'.format(palab)] = (palab in docPalabras)

    for bigramas, i in princBig:
        caracts['Contiene Bigramas: ({})'.format(bigramas)] = (bigramas in bigram)
  
    return caracts

#Listas con las palabras mas comunes
comHam = [(caract(texto), 0) for texto in listadoHam]
comSpam = [(caract(texto), 1) for texto in listadoSpam]
com = comSpam + comHam[:1500]
random.shuffle(com)
#print(comHam, comSpam) #Como se generan los bigramas

# Separacion de listas en entrenar y probar
comEntrenar, comPrueba = train_test_split(com, test_size=0.20, random_state=45)

#Se entrena el programa
classifier = nltk.NaiveBayesClassifier.train(comEntrenar)

#Se prueba y obtiene el nivel de punteria
classifier.classify(caract(listadoHam[34]))
print(nltk.classify.accuracy(classifier, comPrueba))