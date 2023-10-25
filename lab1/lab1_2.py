from nltk.tokenize import word_tokenize
import pandas as pd
from gensim.models import Word2Vec

def load_data():
        return pd.read_csv('lab1/train_en.txt',sep="\t",usecols=["Sentence"]).dropna()

def tokenize(data):
        data['tokens'] = data['Sentence'].apply(lambda x: word_tokenize(x.lower()))

def save(name, words, vectors):
        with open(f'lab1/{name}.txt', 'w+', encoding='utf-8') as doc:
            for word, vector in zip(words, vectors):
                doc.write(word + ' ' + ' '.join(str(value) for value in vector))
                doc.write('\n')

def find_most_similar(word2vec, positive, negative, topn=1):
    similar_words = word2vec.wv.most_similar(positive=positive, negative=negative, topn=topn)
    return similar_words[0][0] if similar_words else None

if __name__== "__main__":
    df = load_data()
    tokenize(df)
    sentences = df['tokens'].values
    word2vec = Word2Vec(sentences, vector_size=100,
                            min_count=2, window=5, sg=1)
    
    #sg=1 skip gram 
    vectors = word2vec.wv.vectors
    id_to_word = word2vec.wv.index_to_key
    save('word2vec', id_to_word, vectors)

    #need this for ex3
    word2vec.save('lab1/word2vec.gensim')

    #lowercase gi napraiv site gore zatoa tuka gi barame vaka
    result1 = find_most_similar(word2vec, positive=['paris', 'italy'], negative=['france'])
    result2 = find_most_similar(word2vec, positive=['madrid', 'france'], negative=['spain'])
    result3 = find_most_similar(word2vec, positive=['king', 'woman'], negative=['man'])
    result4 = find_most_similar(word2vec, positive=['bigger', 'colder'], negative=['big'])
    result5 = find_most_similar(word2vec, positive=['windows','google'], negative=['microsoft'])

    print(f"Paris - France + Italy = {result1}")
    print(f"Madrid - Spain + France = {result2}")
    print(f"King - Man + Woman = {result3}")
    print(f"Bigger - Big + Cold = {result4}")
    print(f"Windows - Microsoft + Google = {result5}")
