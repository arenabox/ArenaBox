import argparse
import json
import time
import timeit
from collections import defaultdict
from os import listdir
from os.path import isfile, join

import hdbscan
import umap
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import clean_text, setup


def create_docs(base_jsonl_folder, eit_json_files, topic_name):
    docs = defaultdict(list)
    eit_json_files = eit_json_files if topic_name is None else [f'{topic_name}.json']
    nlp = setup()

    for eit_json_file in eit_json_files:
        community_name = eit_json_file.split('.')[0]
        with open(join(base_jsonl_folder,eit_json_file), "r") as fd:
            eit_data_json = json.load(fd)
        fd.close()
        print(f'Preprocessing tweets of {community_name}')
        for id, data in tqdm(eit_data_json.items()):
            text = clean_text(data['content'], nlp)
            docs[community_name].append(text)
            docs['all'].append(text)

    return docs

def train_model(docs, topic):
    sentence_model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    start = timeit.default_timer()
    umap_model = UMAP(n_neighbors=15,
                           n_components=10,
                           min_dist=0.0,
                           metric='cosine',)
                           #low_memory=False)
    hdbscan_model = HDBSCAN(min_cluster_size=10,
                                    min_samples=1,
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True)
    model = BERTopic(top_n_words=20,
                     n_gram_range=(1, 3),
                     calculate_probabilities=True,
                     umap_model=umap_model,
                     hdbscan_model=hdbscan_model,
                     verbose=True,
                     nr_topics=30,
                     language="multilingual")
    topics, probabilities = model.fit_transform(docs, embeddings)
    model.save(f"models/{topic}_topic_model")
    end = timeit.default_timer()
    print(f'Total modelling time: {end - start} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for topic modelling')

    parser.add_argument(
        '--topic_name',
        help='Name of the topic to model, if not provided all the topics in dataset will be modelled',
        type=str, required=False, default=None,
    )

    parser.add_argument(
        '--supervised',
        help='If provided then topic modelling is done in supervised manner',
        type=bool, required=False, default=False
    )

    args, remaining_args = parser.parse_known_args()

    base_jsonl_folder = 'data/eit_jsonl'
    eit_json_files = [f for f in listdir(base_jsonl_folder) if
                      isfile(join(base_jsonl_folder, f)) and f.endswith('json')]
    docs = create_docs(base_jsonl_folder=base_jsonl_folder, eit_json_files=eit_json_files, topic_name= args.topic_name)

    if args.topic_name is None:
        train_model(docs=docs['all'], topic='all')
    elif args.topic_name in docs:
        train_model(docs=docs[args.topic_name], topic=args.topic_name)
    else:
        raise ValueError(f'{args.topic_name} is not a valid topic name.\nChoose one from {list(docs.keys())}')

