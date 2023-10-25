#Ova treba ednash pred pochetok
import nltk 
#nltk.download('all')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, WordNetLemmatizer
from nltk.stem import PorterStemmer

if __name__== "__main__":
    sentence = 'Hello bestie. Test sentence number 1!'

    print(sent_tokenize(sentence))
    print(word_tokenize(sentence))
    print(pos_tag(word_tokenize(sentence)))

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    print(stemmer.stem('stripes'))
    print(lemmatizer.lemmatize('stripes','v'))
    print(lemmatizer.lemmatize('stripes','n'))

    print(stemmer.stem('worse'))
    print(lemmatizer.lemmatize('worse'))
    print(lemmatizer.lemmatize('worse','a'))

