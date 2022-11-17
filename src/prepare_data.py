import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from os import listdir
from os.path import join, isfile
from pathlib import Path

import yaml
from tqdm import tqdm

from src.utils import Utils


def break_sentence(long_sentence):
    text = []
    tokens = long_sentence.split()
    if len(tokens) <= 510:
        return long_sentence
    for l in range(0,len(tokens), 510):
        start = l
        end = l + 510
        if end > len(tokens):
            end = len(tokens)
        text.append(' '.join(tokens[start:end]))
    return text


def process_tweets(data, utils):
    text = utils.preprocess_text(data['renderedContent'])
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(data['date'], '%Y-%m-%dT%H:%M:%S+00:00'))
    return text, ts


def process_scientific_articles(data, utils):
    text = []
    abstract = break_sentence(data['abstract'])
    if isinstance(abstract, list):
        text.extend(abstract)
    elif isinstance(abstract, str):
        text.append(abstract)
    for head, content in data['text'].items():
        if content and content != '':
            b = break_sentence(content)
            if isinstance(b,list):
                text.extend(b)
            elif isinstance(b, str):
                text.append(break_sentence(content))
            else:
                raise TypeError(f'Invalid type:{type(b)} returned.')
            #text = text + ' ' + content.strip()
    text = [utils.preprocess_text(t) for t in text]
    year = [datetime.strptime(data['year'], '%Y').year]*len(text)
    return text, year


def create_docs(base_jsonl_folder, json_files, topic_name, type):
    docs = defaultdict(lambda: defaultdict(list))
    json_files = json_files if topic_name == 'all' else [f'{topic_name}.json']
    utils = Utils()
    print('Preprocessing docs')
    for json_file in tqdm(json_files):
        community_name = json_file.split('.')[0]
        with open(join(base_jsonl_folder,json_file), "r") as fd:
            json_data = json.load(fd)
            print(f'Loaded {len(json_data)} data files')
        fd.close()
        for id, data in json_data.items():
            if type == 'tweet':
                text, ts = process_tweets(data, utils)
                docs[community_name]['text'].append(text)
                docs[community_name]['time'].append(ts)  # Used for temporal analysis
                docs[community_name]['class'].append(community_name)  # Used for supervised learning

            elif type == 'sci':
                text, ts = process_scientific_articles(data, utils)
                docs[community_name]['text'].extend(text)
                docs[community_name]['time'].extend(ts)  # Used for temporal analysis
                docs[community_name]['class'].extend([community_name]*len(text))  # Used for supervised learning

            else:
                raise ValueError(f'Invalid type {type}.')

        # POS preprocessing
        tags_to_remove = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']
        docs[community_name]['text'] = utils.pos_preprocessing(docs=docs[community_name]['text'],
                                                               tags_to_remove=tags_to_remove)


        docs['all']['text'] += docs[community_name]['text']
        docs['all']['time'] += docs[community_name]['time']
        docs['all']['class'] += docs[community_name]['class']

    return docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for topic modelling')

    parser.add_argument(
        '-config',
        help='Path to data config file',
        type=str, default='configs/data/base.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    base_jsonl_folder = yaml_data['raw_data_path']
    topic = yaml_data['topic_name'] if yaml_data['topic_name'] is not None else 'all'
    json_files = [f for f in listdir(base_jsonl_folder) if
                      isfile(join(base_jsonl_folder, f)) and f.endswith('json')]
    docs = create_docs(base_jsonl_folder=base_jsonl_folder, json_files=json_files,
                               topic_name=topic, type=yaml_data['type'])
    data_file_path = yaml_data['data_file_path']
    Path(data_file_path).mkdir(parents=True, exist_ok=True)
    with open(f'{data_file_path}/{topic}.json', "w+") as outfile:
        json.dump(docs, outfile, indent=4, sort_keys=False)
    outfile.close()
