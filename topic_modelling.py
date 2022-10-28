import argparse
import json
import time
import timeit
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from pathlib import Path
from random import shuffle
from typing import Optional

from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import Utils


def create_docs(base_jsonl_folder, eit_json_files, topic_name):
    docs = defaultdict(lambda: defaultdict(list))
    eit_json_files = eit_json_files if topic_name is None else [f'{topic_name}.json']
    utils = Utils()
    print('Preprocessing docs')
    for eit_json_file in tqdm(eit_json_files):
        community_name = eit_json_file.split('.')[0]
        with open(join(base_jsonl_folder,eit_json_file), "r") as fd:
            eit_data_json = json.load(fd)
        fd.close()
        for id, data in eit_data_json.items():
            text = utils.preprocess_text(data['content'])
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(data['date'], '%Y-%m-%dT%H:%M:%S+00:00'))
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
    sentence_model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
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
    model = BERTopic(top_n_words=20,
                     n_gram_range=(1, 3),
                     calculate_probabilities=True,
                     umap_model=umap_model,
                     hdbscan_model=hdbscan_model,
                     verbose=True,
                     language="multilingual")
    if supervised:
        model.fit_transform(docs['text'], docs['class'], embeddings)
    else:
        model.fit_transform(docs['text'], embeddings)
    end = timeit.default_timer()
    print(f'Total modelling time: {end - start} seconds')
    return model

def evaluate(docs):
    docs = [doc for doc in docs['text'] if doc != '']
    shuffle(docs)
    tokens = [doc.split() for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = []
    score = []
    for i in range(1, 50):
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
    _ = plt.plot(topics, score)
    _ = plt.xlabel('Number of Topics')
    _ = plt.ylabel('Coherence Score')
    plt.show()
    plt.savefig(f'coherence_plot')


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
    eit_json_files = [f for f in listdir(base_jsonl_folder) if
                      isfile(join(base_jsonl_folder, f)) and f.endswith('json')]
    docs = create_docs(base_jsonl_folder=base_jsonl_folder, eit_json_files=eit_json_files,
                       topic_name=args.topic_name)
    if eval:
        evaluate(docs=docs[topic])
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

