import argparse
import json
from os.path import join
from pathlib import Path

import pandas as pd
import yaml

from sentence_transformers import SentenceTransformer

import collections
from tqdm import tqdm
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
#from cuml.manifold import UMAP
from umap import UMAP
def get_embeddings(embedding_model, data):

  embeddings = embedding_model.encode(data, show_progress_bar=True)
  return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for data preprocessing')

    parser.add_argument(
        '-base_config',
        help='Path to base path config file',
        type=str, default='configs/base_path.yaml',
    )

    parser.add_argument(
        '-config',
        help='Path to data config file',
        type=str, default='configs/data/encode/sci_articles.yaml',
    )
    parser.add_argument(
        '-journal',
        help='Name of data file to preprocess',
        type=str, default=None,
    )



    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    model_name = yaml_data['model_name']

    data_type = yaml_data['type']

    journal = args.journal

    with open(args.base_config) as file:
        base_path = yaml.safe_load(file)
    base_jsonl_folder = join(base_path['data']['preprocess'], data_type)

    file_path = join(base_jsonl_folder, f'{journal}.json')

    json_data = json.load(open(file_path))

    df = pd.DataFrame(json_data)

    data = list(df.text)
    data_file_path = join(base_path['data']['encoded'], data_type)
    Path(data_file_path).mkdir(parents=True, exist_ok=True)
    with open(f'{data_file_path}/only_text.json', 'w+') as f:
        json.dump({'text': data}, f, indent=4)

    # Extract vocab to be used in BERTopic
    vocab = collections.Counter()
    tokenizer = CountVectorizer().build_tokenizer()
    for doc in tqdm(data):
      vocab.update(tokenizer(doc))
    vocab = [word for word, frequency in vocab.items() if frequency >= 15]

    with open(f'{data_file_path}/vocab.txt', 'wb') as fp:
        pickle.dump(vocab, fp)

    embedding_model = SentenceTransformer(model_name)
    embeddings = get_embeddings(embedding_model, data)

    with open(f'{data_file_path}/embeddings.npy', 'wb') as f:
        np.save(f, embeddings)

    # Train model and reduce dimensionality of embeddings
    umap_model = UMAP(n_components=5, n_neighbors=15, random_state=42, metric="cosine", verbose=True)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    with open(f'{data_file_path}/umap_embeddings.npy', 'wb') as f:
        np.save(f, reduced_embeddings)



    # Train model and reduce dimensionality of embeddings
    umap_model = UMAP(n_components=2, n_neighbors=15, random_state=42, metric="cosine", verbose=True)
    reduced_embeddings_2d = umap_model.fit_transform(embeddings)

    with open(f'{data_file_path}/umap_2d_embeddings.npy', 'wb') as f:
        np.save(f, reduced_embeddings_2d)