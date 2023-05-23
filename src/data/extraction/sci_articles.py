import json
from collections import defaultdict
from os.path import splitext, basename, join
from pathlib import Path

import argparse
import yaml

from src.utils.teifile import TEIFile
from src.utils.verify_sci_articles import verify_journal


def tei_to_csv_entry(tei_file, config_data):
    """
    This method creates an instance of the TEIFile class and uses it to parse the TEI file. It uses the information
    provided in config_data to extract specific information from the TEI file and store it in a dictionary called data.
    :param tei_file: name of the TEI file to be parsed
    :param config_data: a dictionary containing information about what information to extract from the TEI file
    :return: the data dictionary containing the extracted information
    """
    tei = TEIFile(tei_file)
    data = defaultdict()
    data['file_name'] = basename_without_ext(tei_file)
    for section in config_data['info_to_extract']:
        if section == 'doi':
            data[section] = tei.doi
        elif section == 'title':
            data[section] = tei.title
        elif section == 'abstract':
            data[section] = tei.abstract
        elif section == 'text':
            data[section] = tei.text
        elif section == 'location':
            data[section] = tei.location
        elif section == 'year':
            data[section] = tei.year
        elif section == 'authors':
            data[section] = tei.authors
        else:
            raise ValueError(f'{section} not found in XML.')
    print(f"Data extracted from {tei_file}")
    return data


def basename_without_ext(path):
    base_name = basename(path)
    stem, ext = splitext(base_name)
    if stem.endswith('.tei'):
        # Return base name without tei file
        return stem[0:-4]
    else:
        return stem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for xml to json convert script')

    parser.add_argument(
        '-config',
        help='Path to data extraction config file for scientific articles',
        type=str, default='configs/data/extraction/sci_articles.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    data_type = config_data['type']
    corpus_name = config_data['corpus_name']
    dir_name = join(base_path['data']['extraction'], data_type)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    xml_path = join(base_path['data']['collection'], data_type, 'xmls', base_path[data_type]['xmls'][corpus_name])
    papers = sorted(Path(xml_path).glob('*.tei.xml'))
    data = defaultdict(dict)
    csv_entries = []
    for paper in papers:
        csv_entries.append(tei_to_csv_entry(paper, config_data))
    data = {i + 1: dat for i, dat in enumerate(csv_entries)}
    if 'metadata_update' in config_data:
        metadata_path = join(base_path['data']['collection'], data_type,'metadata', base_path[data_type]['metadata'][corpus_name])
        data = verify_journal(data, metadata_path)
    with open(join(dir_name, f"{corpus_name}.json"), "w+") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)
