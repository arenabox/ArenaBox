from collections import defaultdict
from os.path import join
from pathlib import Path

import argparse
import json
import numpy as np
import pandas as pd
import yaml
from bertopic import BERTopic

from src.analyze.temporal_trends import get_topic_loadings, fetch_metadata, prepare_full_metadata
from src.utils.graph_plots import create_edge_trace, create_node_trace, create_network_graph, \
    create_within_community_graph


def get_authors(authors_list):
    authors = []
    doc_to_authors = defaultdict(list)
    for i, ats in enumerate(authors_list):
        at = ats.split(',')
        at = [a.strip() for a in at]
        doc_to_authors[i] = at
        authors.extend(at)
    authors = list(set(authors))
    return authors, doc_to_authors

def author2docs(authors, doc_to_authors):
    author_to_doc = defaultdict(list)
    for author in authors:
        temp = []
        for i, al in doc_to_authors.items():
            if author in al:
                temp.append(i)
        author_to_doc[author].extend(temp)
    return author_to_doc

def author2topic(topic_to_docs,author_to_doc):
    author_to_topic = defaultdict(list)
    for author, docs in author_to_doc.items():
        for j, ds in topic_to_docs.items():
            comn = list(set(docs).intersection(ds))
            if len(comn)>0:
                author_to_topic[author].append(j)
    return author_to_topic

def author_to_words(author_to_topic, topic2words):
    author_to_word = defaultdict(list)
    for author, topics in author_to_topic.items():
        vocab = []
        for topic in topics:
            words = topic2words[topic]
            vocab.extend(words)
        author_to_word[author] = list(set(vocab))
    return author_to_word


def author_pipeline(config, name):
    data_type = config['type']
    data_file_path = join(base_path['data']['preprocess'], data_type, f"{name}.json")

    with open(data_file_path, "r") as fd:
        model_data = json.load(fd)
    fd.close()
    model_path = join('models', data_type, f"{name}_topic_model", "model")
    model = BERTopic.load(model_path)

    if 'reduce' in config:
        model.reduce_topics(model_data['text'], config['reduce'][name])

    topics = model.get_topics()
    del topics[-1]  # remove words which are not categorized to any topic
    topic_names = model.get_topic_info()[model.get_topic_info()['Topic'] != -1]['Name']

    topic2words = {t: [word[0] for word in words] for t, words in topics.items()}
    topic_words = [word[0] for t, words in topics.items() for word in words]
    vocab = list(set(topic_words))

    total_docs = max(model_data['id']) + 1
    total_topics = len(topic2words)

    data = pd.DataFrame(model_data)  # Dataframe containing model data
    topic_doc = pd.DataFrame(
        model.probabilities_)  # Dataframe containing topic and their respective probability against each document

    doc_id2idx = defaultdict(list)  # mapping from document id to their respective indices in model_data or data
    if name == 'all':
        title2docs = data.groupby('title').groups
        doc_id2idx = list(title2docs.values())
    else:
        for doc_id in range(0, total_docs):
            idxs = data[data.id == doc_id].index
            doc_id2idx[doc_id] = idxs

    topic_loadings = get_topic_loadings(topic_doc, doc_id2idx, total_docs, config)
    topic2docs = defaultdict(list)  # above 10% loading
    limit = config['temporal']['threshold']
    for i in range(0, total_topics):
        ind = list(topic_loadings.T[topic_loadings.loc[i, :] > limit][i].index)
        topic2docs[i] = ind

    metadata = fetch_metadata(
        join(base_path['data']['collection'], data_type, 'metadata'), base_path[config['type']]['metadata'],
        name)
    metadata = prepare_full_metadata(data, doc_id2idx, total_docs, metadata)

    authors_list = list(metadata['authors'].values())
    authors, doc_to_authors = get_authors(authors_list)
    author_to_doc = author2docs(authors, doc_to_authors)
    author_to_topic = author2topic(topic2docs, author_to_doc)

    return author_to_words(author_to_topic, topic2words), author_to_doc, vocab

def add_author_colab(author_colab, auth_to_words, auth_to_doc):
    for author1, _ in auth_to_words.items():
        docs = auth_to_doc[author1]
        for author2, docss in auth_to_doc.items():
            if author1 == author2 or author2 not in auth_to_words:
                continue
            if len(set(docs).intersection(set(docss))) > 0:
                author_colab[author1].append(author2)
    return author_colab

def get_colab_info(all_authors,subj_a2w, subj2vocab,subj2id):
    author_info = np.zeros((len(all_authors), 5))
    authors_vocab = []
    subj2prop = defaultdict(float)
    for i, author in enumerate(all_authors):
        author_vocab = []
        high_v = 0
        clss = -1
        for subj, idx in subj2id.items():
            if author in subj_a2w[subj]:
                if len(subj_a2w[subj][author]) > high_v:
                    high_v = len(subj_a2w[subj][author])
                    clss = idx
                author_vocab.extend(subj_a2w[subj][author])

        for subj, idx in subj2id.items():
            common_vocab = set(author_vocab).intersection(set(subj2vocab[subj]))
            subj2prop[subj] = len(common_vocab) / len(subj2vocab[subj])

        total = sum(subj2prop.values())
        values = []
        for subj, prop in subj2prop.items():
            subj2prop[subj] = prop/total
            values.append(subj2prop[subj])

        values.extend([len(author_vocab), clss])
        author_info[i] = np.array(values)
        authors_vocab.append(str(author_vocab))

    column_names = list(subj2id.keys()) + ['Vocab Count', 'Class']
    df = pd.DataFrame(author_info, columns=column_names, index=all_authors)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Authors collaborations')

    parser.add_argument(
        '-config',
        help='Path to config file',
        type=str, default='./configs/analyse/colab.yaml',
    )

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    author_colab = defaultdict(list)
    subj_a2w = defaultdict(lambda: defaultdict(list))
    subj2vocab = defaultdict(list)
    all_authors = []
    subj2idx = {subj:i for i, subj in enumerate(sorted(config['network']))}
    for name in list(subj2idx.keys()):
        auth_to_words, auth_to_doc, vocab = author_pipeline(config=config, name=name)
        subj_a2w[name] = auth_to_words
        subj2vocab[name] = vocab
        all_authors.extend(list(auth_to_words.keys()))
        author_colab = add_author_colab(author_colab, auth_to_words, auth_to_doc)
    all_authors = list(set(all_authors))
    all_authors.sort()

    colab_info = get_colab_info(all_authors, subj_a2w, subj2vocab, subj2idx)

    id2class = {i:subj for i, subj in subj2idx.items()}
    class2marker = {0: 'circle', 1: 'circle', 2: 'circle'}
    markers = [class2marker[clss] for clss in list(colab_info['Class'].values)]
    class2color = {0: 'red', 1: 'indigo', 2: "yellowgreen"}
    node_colors = [class2color[clss] for clss in list(colab_info['Class'].values)]
    edgecolormap = {0: "blue", 1: "yellowgreen", 2: "orange", 3: "yellow", 4: 'purple'}

    # Within original Community Collaboration

    plot_path = join(base_path['data']['analysis']['plots'], config['type'])
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    name = join(plot_path,'within_form_colab_network.html')
    title = '<br>Colaboration within original forms'

    G = create_within_community_graph(node_list=all_authors, colab_info=colab_info, edge_info=author_colab,
                                      subjects=list(subj2idx.keys()), edgecolormap=edgecolormap)
    hover_text = list(colab_info.index)
    node_weights = list(colab_info['Vocab Count']*0.15)
    traces = create_edge_trace(G, dim=3)
    traces.append(create_node_trace(G, hover_text=hover_text, node_weights=node_weights, color=node_colors, dim=3))
    create_network_graph(traces, title, name, subjects=list(subj2idx.keys()), dim=3)