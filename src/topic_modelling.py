import argparse
import json
import time
import timeit
from collections import defaultdict
from datetime import datetime
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Optional

import yaml
from bertopic import BERTopic
# Uncomment following for NORMAL computation
from hdbscan import HDBSCAN
from umap import UMAP

# Uncomment following for faster computation
#from cuml.cluster import HDBSCAN
#from cuml.manifold import UMAP

from sentence_transformers import SentenceTransformer


from eval_topic import using_lda
from src.prepare_data import create_docs



def train_model(docs, args):
    supervised = args['supervised']
    sentence_model_args = args['sentence_model']
    dim_reduction_args = args['dim_reduct']
    clustering_args = args['cluster']
    bertopic_args = args['bertopic']
    n_gram_range = tuple(args['n_gram_range'])
    sentence_model = SentenceTransformer(sentence_model_args['name'])
    sentence_model.max_seq_length = sentence_model_args['max_seq_len']
    embeddings = sentence_model.encode(docs['text'], show_progress_bar=True)
    start = timeit.default_timer()
    umap_model = UMAP(**dim_reduction_args['umap'])
                           #low_memory=False)
    hdbscan_model = HDBSCAN(**clustering_args['hdbscan'])
    model = BERTopic(umap_model=umap_model,
                     hdbscan_model=hdbscan_model,
                     n_gram_range=n_gram_range,
                     **bertopic_args
                     )
    if supervised:
        model.fit_transform(documents=docs['text'], y=docs['class'],  embeddings=embeddings)
    else:
        model.fit_transform(docs['text'], embeddings)
    end = timeit.default_timer()
    print(f'Total modelling time: {end - start} seconds')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for topic modelling')

    parser.add_argument(
        '-config',
        help='Path to config file',
        type=str, default='./configs/train/base.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    topic = yaml_data['dataset']['topic_name'] if yaml_data['dataset']['topic_name'] is not None else 'all'

    data_file_path = yaml_data['dataset']['data_file']

    if data_file_path is None:
        dataset_args = yaml_data['dataset']['raw_data']
        base_jsonl_folder = dataset_args['data_path']
        json_files = [f for f in listdir(base_jsonl_folder) if
                          isfile(join(base_jsonl_folder, f)) and f.endswith('json')]
        docs = create_docs(base_jsonl_folder=base_jsonl_folder, json_files=json_files,
                                   topic_name=topic, type=dataset_args['type'])
    else:
        with open(data_file_path, "r") as fd:
            docs = json.load(fd)
        fd.close()

    if yaml_data['eval']:
        #trained_model = train_model(docs=docs['all'], supervised=args.supervised)
        #get_coherence_score(docs['all'], trained_model)
        using_lda(docs=docs[topic]['text'])
    else:
        if topic in docs:
            trained_model = train_model(docs=docs[topic], args=yaml_data['train'])
        else:
            raise ValueError(f'{topic} is not a valid topic name.\nChoose one from {list(docs.keys())}')

        if yaml_data['save']:
            model_path = f"{yaml_data['model_path']}/{topic}_topic_model"
            Path(model_path).mkdir(parents=True, exist_ok=True)
            trained_model.save(f"{model_path}/model")

