import timeit
from random import shuffle

import pandas as pd
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore
from matplotlib import pyplot as plt



def get_coherence_score(docs, topic_model):
    topics = topic_model.topics_

    # Preprocess Documents
    documents = pd.DataFrame({"Document": docs['text'],
                              "ID": range(len(docs['text'])),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    tokens = [analyzer(doc) for doc in cleaned_docs if analyzer(doc) != []]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = []
    for topic in range(len(set(topics)) - 1):
        doc_words = []
        for word, _ in topic_model.get_topic(topic):
            if word != '':
                doc_words.append(word)
        if not doc_words:
            continue
        topic_words.append(doc_words)

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=tokens,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     coherence='c_v',
                                     processes=4)
    score = coherence_model.get_coherence()
    print(f'c_v = {score}')
    return score

def using_bertopic(docs, topic_name, topic_model):
    coherence_list = []
    for num in range(20, 0, -2):
        score = get_coherence_score(docs, topic_model)
        coherence_list.append(score)
    fig = plt.plot(list(range(20,0,-2)), coherence_list)
    fig.savefig(f'coherence_plot_{topic_name}')


def using_lda(docs):
    docs = [doc for doc in docs if doc != '']
    shuffle(docs)
    tokens = [doc.split() for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = []
    score = []
    for i in range(1, 30):
        start = timeit.default_timer()
        print(f'Epoch {i} starts:')
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers=4, passes=10,
                                 random_state=100)
        cm = CoherenceModel(model=lda_model, texts=tokens, corpus=corpus, dictionary=dictionary,
                            coherence='c_v')
        topics.append(i)
        s = cm.get_coherence()
        score.append(s)
        print(f'Coherence: {s}')
        ends = timeit.default_timer() - start
        print(f'Epoch {i} ends after {ends}s')
    plt.figure()
    _ = plt.plot(topics, score)
    _ = plt.xlabel('Number of Topics')
    _ = plt.ylabel('Coherence Score')
    plt.show()
    plt.savefig(f'coherence_plot')


'''def octis_metric(docs):
    topics, trained_model = train_model(docs)
    output = {'topics':topics}
    tokens = [doc.split() for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # Initialize metric
    npmi = Coherence(texts=tokens, topk=10, measure='c_npmi')
    print(npmi.score(output))'''