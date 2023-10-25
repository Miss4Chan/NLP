from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import gensim
import pandas as pd

def load_data():
    return pd.read_csv('lab1/combined.csv')

def load_word2vec_model():
    return gensim.models.Word2Vec.load('lab1/word2vec.gensim')

def calculate_cosine_similarity(vector1, vector2):
    similarity = cosine_similarity([vector1], [vector2])
    return similarity[0][0]

def calculate_euclidean_distance(vector1, vector2):
    #сличност = 1 – растојание.
    distance = euclidean_distances([vector1], [vector2])
    return (1-distance[0][0])

if __name__ == "__main__":
    word2vec = load_word2vec_model()
    df = load_data()

    for index, row in df.iterrows():
        #sekoja redica treba da gi zememe zborovite da gi najdeme vo postoechkiot model i da gi zememe nivnite vektori
        #sporedbata se pravi na vektorite koi shto se generirani vo 1_2
        word1 = row['Word 1']
        word2 = row['Word 2']

        if word2vec.wv.has_index_for(word1) and word2vec.wv.has_index_for(word2):
            vector1 = word2vec.wv.get_vector(word1)
            vector2 = word2vec.wv.get_vector(word2)

            cosine_sim = calculate_cosine_similarity(vector1, vector2)
            euclidean_dist = calculate_euclidean_distance(vector1, vector2)
            
            #printa milioni decimali zatoa e .3f
            print(f'Cosine Similarity between {word1} and {word2}: {cosine_sim:.3f}')
            print(f'Euclidean Distance between {word1} and {word2}: {euclidean_dist:.3f}')
