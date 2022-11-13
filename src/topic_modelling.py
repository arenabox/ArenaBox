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

from bertopic import BERTopic
from hdbscan import HDBSCAN

# Uncomment following for faster computation
#from cuml.cluster import HDBSCAN
#from cuml.manifold import UMAP

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from umap import UMAP

from eval_topic import using_lda
from utils import Utils


def process_tweets(data, utils):
    text = utils.preprocess_text(data['renderedContent'])
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(data['date'], '%Y-%m-%dT%H:%M:%S+00:00'))
    return text, ts

def process_scientific_articles(data, utils):
    text = data['abstract']
    for head, content in data['text'].items():
        if content and content != '':
            text = text + ' ' + content.strip()
    text = utils.preprocess_text(text)
    year = datetime.strptime(data['year'], '%Y').year
    return text, year

def create_docs(base_jsonl_folder, json_files, topic_name, type):
    docs = defaultdict(lambda: defaultdict(list))
    json_files = json_files if topic_name is None else [f'{topic_name}.json']
    utils = Utils()
    print('Preprocessing docs')
    for json_file in tqdm(json_files):
        community_name = json_file.split('.')[0]
        with open(join(base_jsonl_folder,json_file), "r") as fd:
            json_data = json.load(fd)
        fd.close()
        for id, data in json_data.items():
            if type == 'tweet':
                text, ts = process_tweets(data, utils)
            elif type == 'sci':
                text, ts = process_scientific_articles(data, utils)
            else:
                raise ValueError(f'Invalid type {type}.')
            docs[community_name]['text'].append(text)
            docs[community_name]['time'].append(ts)  # Used for temporal analysis
            docs[community_name]['class'].append(community_name)  # Used for supervised learning

        # POS preprocessing
        tags_to_remove = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']
        docs[community_name]['text'] = utils.pos_preprocessing(docs=docs[community_name]['text'],tags_to_remove=tags_to_remove)

        docs['all']['text'] += docs[community_name]['text']
        docs['all']['time'] += docs[community_name]['time']
        docs['all']['class'] += docs[community_name]['class']

    return docs


def train_model(docs, supervised: Optional[bool] = False):
    sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = sentence_model.encode(docs['text'], show_progress_bar=True)
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
    model = BERTopic(n_gram_range=(1, 3),
                     calculate_probabilities=True,
                     umap_model=umap_model,
                     hdbscan_model=hdbscan_model,
                     verbose=True,
                     nr_topics=12,
                     language="multilingual")
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
        '-data',
        help='Path to json data folder',
        type=str, default='data/eit_jsonl',
    )

    parser.add_argument(
        '-topic_name',
        help="Name of the topic to model, if not provided all the topics in dataset will be modelled.\n"
             "List of Topics: ['ClimateKIC', 'EIT_Digital', 'InnoEnergyEU', 'EITeu', 'EITFood', 'EITHealth', "
             "'EITManufactur','EITUrbanMob', 'EITRawMaterials', 'EU_Commission', 'EUCouncil', 'EUeic', 'europarl'].",
        type=str, required=False, default=None,
    )

    parser.add_argument(
        '-type',
        help='Type of data for topic modelling. Currently, twitter ("tweet") data and scientific("sci") articles included.',
        required=False, type=str, default='twitter',
    )

    parser.add_argument(
        '-supervised',
        help='If provided then topic modelling is done in supervised manner',
        required=False, action='store_true'
    )

    parser.add_argument(
        '-save',
        help='Boolean parameter to decide whether to save trained model and respective stats or not',
        required=False, action='store_true'
    )

    parser.add_argument(
        '-eval',
        help='Perform coherence test on series of topic numbers',
        required=False, action='store_true'
    )


    args, remaining_args = parser.parse_known_args()

    base_jsonl_folder = args.data
    topic = args.topic_name if args.topic_name is not None else 'all'
    json_files = [f for f in listdir(base_jsonl_folder) if
                      isfile(join(base_jsonl_folder, f)) and f.endswith('json')]
    docs = create_docs(base_jsonl_folder=base_jsonl_folder, json_files=json_files,
                               topic_name=args.topic_name, type=args.type)
    if args.eval:
        #trained_model = train_model(docs=docs['all'], supervised=args.supervised)
        #get_coherence_score(docs['all'], trained_model)
        using_lda(docs=docs[topic]['text'])
    else:
        if args.topic_name is None:
            trained_model = train_model(docs=docs['all'], supervised=args.supervised)
        elif args.topic_name in docs:
            trained_model = train_model(docs=docs[args.topic_name], supervised=args.supervised)
        else:
            raise ValueError(f'{args.topic_name} is not a valid topic name.\nChoose one from {list(docs.keys())}')

        if args.save:
            Path(f"models/{topic}_topic_model").mkdir(parents=True, exist_ok=True)
            trained_model.save(f"models/{topic}_topic_model/model")

