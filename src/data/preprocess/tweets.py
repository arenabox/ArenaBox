import json
import time
from collections import defaultdict
from os import listdir
from os.path import join, isfile
from pathlib import Path

import argparse
import yaml
from tqdm import tqdm

from src.utils.data_preprocessor import Preprocessor


def process_tweets(data, preprocessor, config_data):
    content_field = config_data['content_field']
    text = preprocessor.preprocess_text(data[content_field])
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(data['date'], '%Y-%m-%dT%H:%M:%S+00:00'))
    return text, ts


def for_user(file_path, user, config_data):
    """
    This method loads a json file for a particular user, preprocess the data and returns the preprocessed
    data. It uses the Preprocessor class to do the preprocessing and applies the POS preprocessing based on the
    remove_pos key in the config_data. The returned data is a dictionary containing text, time, and class key and
    respective values. text is list of preprocessed tweets, time represent time of each tweet and class is username.

    :param file_path: string representing the file path of a JSON file containing the data to be preprocessed
    :param user: string representing a username/class name for the data
    :param config_data: dictionary containing various configuration options
    :return: a dictionary containing text, time, and class key
    """
    docs = defaultdict(list)
    with open(file_path, "r") as fd:
        json_data = json.load(fd)
        print(f'Loaded {len(json_data)} data files')
    fd.close()
    preprocessor = Preprocessor(config_data['preprocess'])
    print('Preprocessing docs')
    for i, data in tqdm(json_data.items()):
        text, ts = process_tweets(data, preprocessor, config_data)
        docs['text'].append(text)
        docs['time'].append(ts)  # Used for temporal analysis
        docs['class'].append(user)  # Used for supervised learning

    # POS preprocessing
    if 'remove_pos' in config_data['preprocess']:
        docs['text'] = preprocessor.pos_preprocessing(docs=docs['text'])

    return docs


def for_all_users(user_files, config_data):
    """
    This function preprocess tweets for multiple users. It loops through a list of users provided in user_files, calls
    the for_user function for each file and appends the returned data to a single dictionary which is returned at the
    end of the function.

    :param user_files:  list of strings representing the file names of JSON files containing the data to be preprocessed.
    :param config_data: dictionary containing various configuration options
    :return: dictionary, which contains the preprocessed data for all users.
    """
    docs = defaultdict(lambda: defaultdict(list))
    for user_file in tqdm(user_files):
        user = user_file.split('.')[0]
        file_path = join(config_data['raw_data_path'], user_file)
        docs[user] = for_user(file_path, user, config_data)

    return docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for data preprocessing')

    parser.add_argument(
        '-config',
        help='Path to data config file',
        type=str, default='configs/data/preprocess/tweets.yaml',
    )
    parser.add_argument(
        '-user',
        help='Name of data file to preprocess',
        type=str, default=None,
    )

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    data_type = yaml_data['type']

    base_jsonl_folder = join(base_path['data']['collection'], data_type)
    user = args.user

    if user is not None:
        file_path = join(base_jsonl_folder, f'{user}.json')
        docs = for_user(file_path, user, yaml_data)
    else:
        user = 'all'
        user_files = [f for f in listdir(base_jsonl_folder) if
                      isfile(join(base_jsonl_folder, f)) and f.endswith('json')]
        docs = for_all_users(user_files, yaml_data)

    data_file_path = join(base_path['data']['preprocess'], data_type)
    Path(data_file_path).mkdir(parents=True, exist_ok=True)
    with open(f'{data_file_path}/{user}.json', "w+") as outfile:
        json.dump(docs, outfile, indent=4, sort_keys=False)
    outfile.close()
