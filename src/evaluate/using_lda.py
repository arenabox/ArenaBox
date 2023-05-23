import json
from os.path import join
from pathlib import Path
from random import shuffle

import argparse
import numpy as np
import yaml
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore
from matplotlib import pyplot as plt
from tqdm import tqdm


def find_optimal_num_of_topic(docs, config, plot_path):
    """
    This method can be used to find optimal number of topics for given dataset. It uses LDA method to perform topic
    modelling on given range of topics and then find coherence score for resulting model. It returns the topic number
    which results in the highest coherence score.
    :param docs: list of string
    :param config: dictionary containing parameter values
    :return: integer represent optimal number of topics
    """
    docs = [doc for doc in docs if doc != '']
    shuffle(docs)
    tokens = [doc.split() for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = []
    score = []
    for i in tqdm(range(config['optimal_topics']['min_topics'], config['optimal_topics']['max_topics'], config['optimal_topics']['step'])):
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers=4, passes=10,
                                 random_state=100)
        cm = CoherenceModel(model=lda_model, texts=tokens, corpus=corpus, dictionary=dictionary,
                            coherence='c_v')
        topics.append(i)
        s = cm.get_coherence()
        score.append(s)
    plt.figure()
    _ = plt.plot(topics, score)
    _ = plt.xlabel('Number of Topics')
    _ = plt.ylabel('Coherence Score')
    plt.show()
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    plot_path = join(plot_path, f'{config["name"]}_optimal_topics_lda.jpg')
    plt.savefig(plot_path)
    optimal_topic_num = topics[np.argmax(np.array(score))]
    return optimal_topic_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for LDA evaluation')

    parser.add_argument(
        '-config',
        help='Path to config file',
        type=str, default='./configs/evaluate/lda.yaml',
    )

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    data_type = config['type']
    data_file_path = join(base_path['data']['preprocess'], data_type, f"{config['name']}.json")
    with open(data_file_path, "r") as fd:
        docs = json.load(fd)
    fd.close()
    plot_path = join(base_path['data']['analysis']['plots'],data_type)
    optimal_topics = find_optimal_num_of_topic(docs=docs['text'], config=config, plot_path=plot_path)
    print(f'Optimal number of topics for given dataset is {optimal_topics}')
