import json
from collections import defaultdict
from datetime import datetime
from os import listdir
from os.path import join, isfile
from pathlib import Path

import argparse
import yaml
from tqdm import tqdm

from src.utils.data_preprocessor import Preprocessor


def break_sentence(long_sentence, max_len, min_len):
    """
    This method breaks a long sentence into small chunks based on the max_len and min_len.
    :param long_sentence: string representing a sentence
    :param max_len: integer value representing maximum length a chunk can have
    :param min_len: integer value representing minimum length a chunk should have
    :return: a string if original sentence is within given range otherwise list of string representing small chunks of
    long sentence
    """
    text = []
    tokens = long_sentence.split()
    if len(tokens) <= max_len and len(tokens) >= min_len:
        return long_sentence
    for l in range(0,len(tokens), max_len):
        start = l
        end = l + max_len
        if end > len(tokens):
            end = len(tokens)
        if end-start < min_len:
            continue
        text.append(' '.join(tokens[start:end]))
    return text
def process_scientific_article(data, preprocessor, config_data):
    """
    This method is used for processing scientific articles. It checks if there is a "chunk" key in the config_data. If
    there is not, it concatenates the abstract
    and text of the article, preprocesses the text using the preprocessor, and assigns the year of publication to the
    variable "year". If there is a "chunk" key, it breaks the abstract and text of the article into chunks of text using
     the "break_sentence" function and the maximum and minimum length of chunks specified in the config_data. The
     preprocessed chunks are added to the "text" list and the year is repeated and added to the year list. The method
     returns the text and year.
    :param data: a dictionary containing information about the scientific article
    :param preprocessor: an object that contains methods for preprocessing text, such as removing punctuation or stop words.
    :param config_data: a dictionary that contains configuration information for the method
    :return: preprocessing text and year of the article published
    """
    text = []

    if 'chunk' not in  config_data:
        text = data['abstract']
        for head, content in data['text'].items():
            if content and content != '':
                text = text + ' ' + content.strip()
        text = preprocessor.preprocess_text(text)
        year = datetime.strptime(data['year'], '%Y').year
    else:
        max_len = config_data['chunk']['max_seq_len']
        min_len = config_data['chunk']['min_seq_len']
        abstract = preprocessor.preprocess_text(data['abstract'])
        abstract = break_sentence(abstract, max_len = max_len-2, min_len=min_len)
        if isinstance(abstract, list):
            text.extend(abstract)
        elif isinstance(abstract, str):
            text.append(abstract)
        for head, content in data['text'].items():
            if content and content != '':
                content = preprocessor.preprocess_text(content)
                b = break_sentence(content, max_len = max_len-2, min_len=min_len)
                if isinstance(b,list):
                    text.extend(b)
                elif isinstance(b, str):
                    text.append(b)
                else:
                    raise TypeError(f'Invalid type:{type(b)} returned.')
        year = [datetime.strptime(data['year'], '%Y').year]*len(text)

    return text, year


def for_journal(file_path, journal, config_data):
    """
    This method loads a json file for a particular journal, preprocess the data and returns the preprocessed
    data. It uses the Preprocessor class to do the preprocessing and applies the POS preprocessing based on the
    remove_pos key in the config_data. This method allows dividing large data into small chunks so that full text
    can be processed by language models. The returned data is a dictionary containing title, abstract, text, time, class,
    id key and respective values. text is list of preprocessed content of the article, time represent the year of the
    article published, class represent the journal name of the article and id is unique number given to the article.
    If we divide document into small chunks then chunk of same article gets same id.

    :param file_path: string representing the file path of a JSON file containing the data to be preprocessed
    :param user: string representing a journal/class name for the data
    :param config_data: dictionary containing various configuration options
    :return: a dictionary containing title, abstract, text, time, id and class key
    """
    docs = defaultdict(list)
    with open(file_path, "r") as fd:
        json_data = json.load(fd)
        print(f'Loaded {len(json_data)} data files')
    fd.close()
    preprocessor = Preprocessor(config_data['preprocess'])
    print('Preprocessing docs')
    id = 0
    for i, data in json_data.items():
        text, ts = process_scientific_article(data, preprocessor, config_data)
        if 'chunk' in config_data:
            docs['title'].extend([data['title']] * len(text))
            docs['abstract'].extend([data['abstract']] * len(text))
            docs['text'].extend(text)
            docs['id'].extend([id] * len(text))
            docs['time'].extend(ts)  # Used for temporal analysis
            docs['class'].extend([journal] * len(text))  # Used for supervised learning
        else:
            docs['text'].append(text)
            docs['abstract'].append(data['abstract'])
            docs['time'].append(ts)  # Used for temporal analysis
            docs['id'].append(id)  # Used for supervised learning
            docs['class'].append(journal)
            docs['title'].append(data['title'])

        id += 1

    # POS preprocessing
    if 'remove_pos' in config_data['preprocess']:
        docs['text'] = preprocessor.pos_preprocessing(docs=docs['text'])
    return docs


def for_all_journals(journal_files, config_data, base_json_folder):
    """
    This function preprocess scientific articles for multiple journals. It loops through a list of journals provided in
    journal_files, calls the for_journal function for each file and appends the returned data to a single dictionary
    which is returned at the end of the function.

    :param journal_files:  list of strings representing the file names of JSON files containing the data to be preprocessed.
    :param config_data: dictionary containing various configuration options
    :return: dictionary, which contains the preprocessed data for all journals.
    """
    docs =  defaultdict(list)
    for journal_file in tqdm(journal_files):
        journal = journal_file.split('.')[0]
        file_path = join(base_json_folder, journal_file)
        for k,v in for_journal(file_path, journal, config_data).items():
            docs[k].extend(v)
    return docs


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
        type=str, default='configs/data/preprocess/sci_articles.yaml',
    )
    parser.add_argument(
        '-journal',
        help='Name of data file to preprocess',
        type=str, default=None,
    )



    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    data_type = yaml_data['type']

    journal = args.journal


    with open(args.base_config) as file:
        base_path = yaml.safe_load(file)
    base_jsonl_folder = join(base_path['data']['extraction'], data_type)



    if journal is not None:
        file_path = join(base_jsonl_folder, f'{journal}.json')
        docs = for_journal(file_path, journal, yaml_data)
    else:
        journal = 'all'
        journal_files = [f for f in listdir(base_jsonl_folder) if
                      isfile(join(base_jsonl_folder, f)) and f.endswith('json')]
        docs = for_all_journals(journal_files, yaml_data, base_jsonl_folder)

    data_file_path = join(base_path['data']['preprocess'], data_type)
    Path(data_file_path).mkdir(parents=True, exist_ok=True)
    with open(f'{data_file_path}/{journal}.json', "w+") as outfile:
        json.dump(docs, outfile, indent=4, sort_keys=False)
    outfile.close()
