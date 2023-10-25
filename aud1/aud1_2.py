#word vec na word occurences -- frequencies of appearing together
#GloVe -- puts together learning and frequencies
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

def load_data():
    return pd.read_csv('/Users/despinamisheva/Desktop/NLP Aud i Lab/covid19_tweets.csv',usecols=['text']).dropna()

def tokenize(data):
    data['tweet_tokens'] = data['text'].apply(lambda x: word_tokenize(x.lower()))

def save(name, words, vectors):
    with open(f'{name}.txt', 'w+', encoding='utf-8') as doc:
        for word, vector in zip(words, vectors):
            doc.write(word + ' ' + ' '.join(str(value) for value in vector))
            doc.write('\n')

if __name__== "__main__":
    df = load_data()
    df = df.head(100)
    tokenize(df)
    sentences = df['tweet_tokens'].values
    word2vec = Word2Vec(sentences, vector_size=45,
                         min_count=10, window=3,sg=1)
    
    #sg=1 skip gram, sg=2 countinous bag of words
    vectors = word2vec.wv.vectors
    id_to_word = word2vec.wv.index_to_key
    save('word2vec', id_to_word, vectors)