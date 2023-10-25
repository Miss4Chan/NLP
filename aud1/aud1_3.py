from gensim import downloader

if __name__ == '__main__':
    embeddings = downloader.load('glove-twitter-50')

    print(embeddings.most_similar('student'))

    # Paris - France + Italy
    print(embeddings.most_similar(positive=['paris', 'italy'], negative=['france']))

    # King - Man + Woman
    print(embeddings.most_similar(positive=['king', 'woman'], negative=['man']))