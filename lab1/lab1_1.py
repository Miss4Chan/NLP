from nltk.tokenize import word_tokenize
from nltk import FreqDist, WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd

def load_data():
    return pd.read_csv('lab1/train_en.txt',sep="\t",usecols=["Sentence"]).dropna()

def tokenize(data):
    data['tokens'] = data['Sentence'].apply(lambda x: word_tokenize(x.lower()))

def getAllWords(df):
    #za sekoi tokens vo df za sekoj token vo tokens
    return [token for tokens in df['tokens'] for token in tokens]

def calculateFreqAndUnique(allWords):
    vocabSet = set(allWords)
    freqDist = FreqDist(allWords)
    return len(vocabSet),freqDist

def calculateNoPunc(allWords):
    #site unikatni zborovi vo rechenicive
    stopWords = stopwords.words('english')
    punctuationList = list(punctuation)

    wordsWithoutStopsAndPunc = [token for token in allWords if token not in stopWords and token not in punctuationList]

    vocabSetNoPunc, freqDistNoPunc = calculateFreqAndUnique(wordsWithoutStopsAndPunc)
    return vocabSetNoPunc,freqDistNoPunc, wordsWithoutStopsAndPunc;

def findLemVocab(cleanWords):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in cleanWords]

def findStemVocab(cleanWords):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in cleanWords]

def printFrequency(wordFreq):
    #item= item[1] za da sortira po vtorata var u parot aka po freq
    #sorted za da printam samo 20 najchesti :)
    for token, frequency in sorted(wordFreq.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"Token: {token}, Frequency: {frequency}")

if __name__== "__main__":
    df = load_data();
    tokenize(df)

    allWords = getAllWords(df)
    totalCount = len(allWords)
    uniqueTokens,tokenFrequencies = calculateFreqAndUnique(allWords)
    uniqueWords, wordFrequencies, cleanWords = calculateNoPunc(allWords)

    print(totalCount)
    printFrequency(tokenFrequencies)
    print(uniqueTokens)
    print()

    print(wordFrequencies)
    printFrequency(wordFrequencies)
    print(uniqueWords)

    lemWords = findLemVocab(cleanWords)
    uniqueLem, freqLem = calculateFreqAndUnique(lemWords)
    print(uniqueLem)
    printFrequency(freqLem)

    stemWords = findStemVocab(cleanWords)
    uniqueStem, freqStem = calculateFreqAndUnique(stemWords)
    print(uniqueStem)
    printFrequency(freqStem)