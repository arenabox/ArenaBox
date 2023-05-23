import json
import timeit
from os.path import join
from pathlib import Path

import argparse
import yaml
from bertopic import BERTopic
# Uncomment following for NORMAL computation
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from src.evaluate.using_lda import find_optimal_num_of_topic


# Uncomment following for faster computation
# from cuml.cluster import HDBSCAN as fHDBSCAN
# from cuml.manifold import UMAP as fUMAP


def get_embeddings(docs,args):
    """
    It uses a transformer model to get embeddings for documents in docs. Currently, only sentence transformer is supported
    but ST can take different pre-trained models as listed here: https://www.sbert.net/docs/pretrained_models.html
    :param docs: dictionary containing document as value of 'text' key
    :param args: dictionary containing parameter values
    :return: numpy array containing embedding for all documents
    """
    model_type = args['use']
    if model_type == 'sentence_transformer':
        embedding_model = SentenceTransformer(args['name'])
        embedding_model.max_seq_length = args['max_seq_len']
        embeddings = embedding_model.encode(docs['text'])
    else:
        raise ValueError(f'Given type: {model_type} not found.')

    return embeddings

def train_model(docs, config):
    """
    This method train a topic model based on config values provided. Bertopic is used to modelling, which is embedding
    based topic modelling technique and provide multiple ways to do it. For more info see here: https://github.com/MaartenGr/BERTopic
    Different embedding models can be used which are defined in config, parameters for clustering and dimensionality
    reduction method can also be varied using config. It also allows supervised modelling which requires data to have
    class for each document prior to modelling.
    :param docs: dictionary containing document as value of 'text' key
    :param config: dictionary containing parameter values
    :return: trained topic model
    """
    supervised = config['supervised']
    embedding_model_args = config['embedding_model']
    dim_reduction_args = config['dim_reduct']
    clustering_args = config['cluster']
    bertopic_args = config['bertopic']
    n_gram_range = tuple(config['n_gram_range'])

    print('Converting text to vectors ...')
    embeddings = get_embeddings(docs=docs, args=embedding_model_args)

    start = timeit.default_timer()
    umap_model = UMAP(random_state=42,**dim_reduction_args['umap'])
    # low_memory=False)
    hdbscan_model = HDBSCAN(**clustering_args['hdbscan'])
    model = BERTopic(umap_model=umap_model,
                         hdbscan_model=hdbscan_model,
                         n_gram_range=n_gram_range,
                         **bertopic_args
                         )
    print(f'Starting modelling ... ')
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

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    data_type = config['type']
    data_file_path = join(base_path['data']['preprocess'], data_type, f"{config['name']}.json")

    model_path = join(base_path['model'],data_type)

    with open(data_file_path, "r") as fd:
        docs = json.load(fd)
    fd.close()

    if 'find_optimal_topic_first' in config:
        with open(config['find_optimal_topic_first']['config_path']) as file:
            ot_config = yaml.safe_load(file)

        plot_path = join(base_path['data']['analysis']['plots'], data_type)
        num_topics = find_optimal_num_of_topic(docs=docs['text'], config=ot_config, plot_path=plot_path)
        config['train']['bertopic']['nr_topics'] = num_topics
    trained_model = train_model(docs=docs, config=config['train'])

    if config['save']:
        model_path = f"{model_path}/{config['name']}_topic_model"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        trained_model.save(f"{model_path}/model")

