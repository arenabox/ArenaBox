import json
from collections import defaultdict, Counter
from os.path import join
from pathlib import Path

import argparse
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from bertopic import BERTopic

from src.analyze.temporal_trends import get_topic_loadings, fetch_metadata
from src.utils.graph_plots import create_random_graph, create_edge_trace, create_node_trace, create_network_graph, \
    create_pruned_graph


def get_edge_data(topic_to_docs):
    # Edge Data
    edge_data = defaultdict(dict)
    for topic_nr, docs in topic_to_docs.items():
        for topic_nr2, docs2 in topic_to_docs.items():
            if topic_nr == topic_nr2:
                continue
            commn = set(docs).intersection(set(docs2))
            if len(commn)!=0:
                edge_data[topic_nr][topic_nr2] = len(commn)
    return edge_data


def pruned_full_network(plot_path, threshold):
    # Topic Network with weights greater than 1
    G, node_colors, node_weights = create_pruned_graph(topics, node_data, edge_data, thres=threshold)
    name = join(plot_path, f'topic_network_3_journals_antons_wgt{threshold}+.html')
    title = f'<br>Topic Network with weights greater than {threshold}'

    hover_text = [f'{tn}: {words}' for tn, words in topic2words.items()]
    traces = create_edge_trace(G)
    traces.append(create_node_trace(G, hover_text, node_weights, color=node_colors))
    create_network_graph(traces, title, name)

def get_node_weights(topic_loadings, num_topics):
    # Node Data
    node_weights = []
    for i in range(0,num_topics):
        node_weights.append(topic_loadings.loc[i,:].sum())
    return node_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Topic network visualization')

    parser.add_argument(
        '-config',
        help='Path to config file',
        type=str, default='./configs/analyse/topic_networks.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    with open(args.config) as file:
        config = yaml.safe_load(file)

    data_type = config['type']
    data_path = join(base_path['data']['preprocess'], data_type, f"{config['name']}.json")

    metadata = fetch_metadata(
    join(base_path['data']['collection'], data_type, 'metadata'), base_path[config['type']]['metadata'], config['name'])

    plot_path = join(base_path['data']['analysis']['plots'],data_type)
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    model_path = join('models', data_type, f"{config['name']}_topic_model", "model")
    model = BERTopic.load(model_path)
    with open(data_path, "r") as fd:
        model_data = json.load(fd)
    if 'reduce' in config:
        model.reduce_topics(model_data['text'], config['reduce'][config['name']])

    topics = model.get_topics()
    del topics[-1] # remove words which are not categorized to any topic
    topic_names = model.get_topic_info()[model.get_topic_info()['Topic']!=-1]['Name']

    topic2words = {t: [word[0] for word in words] for t, words in topics.items()}
    topic_words = [word[0] for t, words in topics.items() for word in words]
    total_docs = max(model_data['id']) + 1
    total_topics = len(topic2words)

    data = pd.DataFrame(model_data) # Dataframe containing model data
    topic_doc = pd.DataFrame(model.probabilities_) # Dataframe containing topic and their respective probability against each document

    doc_id2idx = defaultdict(list)  # mapping from document id to their respective indices in model_data or data

    if config['name'] == 'all':
        title2docs = data.groupby('title').groups
        total_docs = len(title2docs)
        for doc_id, (title, idxs) in enumerate(title2docs.items()):
            doc_id2idx[doc_id] = idxs
    else:
        for doc_id in range(0, total_docs):
            idxs = data[data.id==doc_id].index
            doc_id2idx[doc_id] = idxs
    clsses = defaultdict()
    for i, idx in doc_id2idx.items():
        clss = data.loc[idx, 'class'].unique()
        if len(clss) == 1:
            clsses[i] = clss[0]

    topic_loadings = get_topic_loadings(topic_doc, doc_id2idx, total_docs, config, plot_path)

    topic_to_docs = defaultdict(list)  # above 10% loading
    limit = config['temporal']['threshold']
    topic_to_cls = defaultdict(list)
    for i in range(0, total_topics):
        ind = list(topic_loadings.T[topic_loadings.loc[i, :] > limit][i].index)
        topic_to_docs[i] = ind
        topic_to_cls[i] = [clsses[j] for j in ind]


    if config['name'] == 'all':
        node_data = np.zeros((len(topic_to_cls), 4))
        for topic_nr, journals in topic_to_cls.items():
            dt = Counter(journals)
            subj_prop = defaultdict(float)
            for subj, count in dict(dt).items():
                subj_prop[subj] = count/sum(dt.values())
            values = list(subj_prop.values())
            values.extend([sum(dt.values())])
            node_data[topic_nr] = np.array(values)

        edge_data = defaultdict(dict)
        for topic_nr, docs in topic_to_docs.items():
            for topic_nr2, docs2 in topic_to_docs.items():
                if topic_nr == topic_nr2:
                    continue
                commn = set(docs).intersection(set(docs2))
                if len(commn) != 0:
                    edge_data[topic_nr][topic_nr2] = len(commn)

        G = nx.random_geometric_graph(len(topics), 0.125)
        G.remove_edges_from(list(G.edges()))
        node_list = list(topics.keys())
        node_colors = []
        node_weights = []
        for node in node_list:
            code = [255, 255, 255]
            pos = node_data[node][:3]
            node_colors.append(f"rgb{tuple(np.array(code * pos, dtype=int))}")
            node_weights.append(node_data[node][3])

        for src, tgts in edge_data.items():
            for tgt, wgt in tgts.items():
                G.add_edge(src, tgt, weight=wgt)
        name = join(plot_path,f'topic_network_3_journals_antons.html')
        title = '<br>Topic Network for 3 journals'

        hover_text = [f'{tn}: {words}' for tn, words in topic2words.items()]
        traces = create_edge_trace(G)
        traces.append(create_node_trace(G, hover_text, node_weights, color=node_colors))
        create_network_graph(traces, title, name)

        # Topic Network with weights greater than threshold
        for threshold in config['pruned_graph']:
            pruned_full_network(plot_path, threshold=threshold)
    else:
        name = join(plot_path,f'topic_network_{config["name"].lower()}_antons.html')
        title = f'<br> {config["name"]} Topic Network'
        node_weights = get_node_weights(topic_loadings, num_topics=total_topics)
        edge_data = get_edge_data(topic_to_docs)
        G = create_random_graph(num_of_nodes=total_topics, edge_data=edge_data)
        hover_text = [f'{tn}: {words}' for tn, words in topic2words.items()]
        traces = create_edge_trace(G, dim=2)
        traces.append(create_node_trace(G, hover_text, node_weights, color=config['colormap'][config['name']], dim=2))
        create_network_graph(traces, title, name, dim=2)


