import argparse

import pandas as pd
import plotly.graph_objects as go

import networkx as nx
from bertopic import BERTopic
from pyvis.network import Network

def create_graph_data(topics):
    data = {'source':[], 'target':[], 'weight':[]}
    for k, v in topics.items():
        if k==-1:
            continue
        for word, prob in v:
            data['source'].append(f'Topic {k+1}')
            data['target'].append(word)
            data['weight'].append(prob)
    return data

def draw_graph(data):
    df = pd.DataFrame(data=data)
    net = Network(height="100%", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    sources = df['source']
    targets = df['target']
    #weights = df['weight']

    edge_data = zip(sources, targets)#, weights)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        #w = e[2]

        net.add_node(src, src, title=src,group=src)
        net.add_node(dst, dst, title=dst,group=src)
        net.add_edge(src, dst, )#value=w)

    neighbor_map = net.get_adj_list()

    # add neighbor data to node hover data
    for node in net.nodes:
        node["title"] += " Neighbors: " + " ".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])
    return net

def draw_graph_2(data):
    df = pd.DataFrame(data=data)
    # Custom Graph
    net = Network(height="100%", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    node_list = set(df['target'].values).union( set(df['source'].values))
    for node in set(df['source'].values):
        net.add_node(node,size=25,group=node,title=node)
    sources = df['source']
    targets = df['target']
    #weights = df['weight']

    edge_data = zip(sources, targets)
    for e in edge_data:
        src = e[0]
        dst = e[1]
        net.add_node(dst,size=10, group=src,title=dst)
        net.add_edge(src,dst)
    neighbor_map = net.get_adj_list()

    # add neighbor data to node hover data
    for node in net.nodes:
        node["title"] += " Neighbors: " + " ".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for topic modelling')

    parser.add_argument(
        '-model_path',
        help='Path to json data folder',
        type=str, required=True,
    )
    parser.add_argument(
        '-graph_name',
        help='Path to json data folder',
        type=str, default='test',
    )

    args, remaining_args = parser.parse_known_args()

    loaded_model = BERTopic.load(args.model_path)
    topics = loaded_model.get_topics()
    data = create_graph_data(topics)
    net = draw_graph_2(data)
    net.show(args.graph_name)