import argparse
import json
from os.path import join

import yaml

from src.evaluate.using_lda import find_optimal_num_of_topic
from topic_modelling import TopicModelling


# Uncomment following for NORMAL computation
# Uncomment following for faster computation
# from cuml.cluster import HDBSCAN as fHDBSCAN
# from cuml.manifold import UMAP as fUMAP



def get_labels(topic_model):
    labels = []
    for i in list(topic_model.get_topic_info()['Llama2']):
        t = i[0].split('\n')
        if len(t) == 1:
            labels.append(t[0])
        else:
            if t[0] != '':
                labels.append(t[0])
            else:
                if t[1].startswith('Label:'):
                    p = t[1].split('Label:')[1]
                    labels.append(p)
                else:
                    labels.append(t[1])
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for topic modelling')

    parser.add_argument(
        '-base_config',
        help='Path to base path config file',
        type=str, default='configs/base_path.yaml',
    )

    parser.add_argument(
        '-config',
        help='Path to data config file',
        type=str, default='configs/train/core.yaml',
    )
    parser.add_argument(
        '-journal',
        help='Name of data file to model',
        type=str, default=None,
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    data_type = config['type']
    journal = args.journal

    with open(args.base_config) as file:
        base_path = yaml.safe_load(file)
    base_jsonl_folder = join(base_path['data']['encoded'], data_type)


    if 'find_optimal_topic_first' in config:
        with open(config['find_optimal_topic_first']['config_path']) as file:
            ot_config = yaml.safe_load(file)

        docs = json.load(open(f'{base_jsonl_folder}/only_text.json'))

        plot_path = join(base_path['data']['analysis']['plots'], data_type)
        num_topics = find_optimal_num_of_topic(docs=docs['text'], config=ot_config, plot_path=plot_path)
        config['train']['bertopic']['nr_topics'] = num_topics

    tp = TopicModelling(base_path=base_jsonl_folder, config=config)
    topic_model = tp.get_topic_model()
    llama2_labels = get_labels(topic_model)
    topic_model.set_topic_labels(llama2_labels)
    if config['save']:
        tp.save_model()

