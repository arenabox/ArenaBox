import json
import logging
from collections import defaultdict

import argparse
import pandas as pd


def filename_to_doi(filename):
    filename = '/'.join(filename.split('_'))
    parts = filename.split('-')
    part1 = parts[0]
    rest = parts[1:]
    filename = part1+'.'+'-'.join(rest)
    return filename

def update_data(journal_data,metadata):
    """
    This method updates journal_data based on the metadata. It looks for DOI in the metadata for each extracted doi. If
    DOI is found then it compares title, abstract and year and updates if there is mismatch. We consider metadata to have
    correct information, so we use it for updating in case of a mismatch.
    We keep a stats of doi which was in metadata but not in the journal_data i.e. it was not found during data extraction
    process. Similarly, we keep track of all dois which were in journal_data but not in metadata. This stats can be used
    to fix the dataset manually later if required.

    :param journal_data: dictionary containing extract information for each article
    :param metadata: a dictionary mapping doi to (abstract, title and year) of an article
    :return: updated data
    """
    new_data = defaultdict(defaultdict)
    xml_dois = []
    common_dois = []
    stats = {'doi_not_in_xml': [],
             'doi_not_in_original': []}
    for id, d in journal_data.items():
        doi = d['doi'].lower() # enable this for sus_sci corpus  .split('(')[0].split(')')[0]
        filename = filename_to_doi(d['file_name'].lower())
        xml_dois.append(doi)
        if doi in metadata.keys() :
            if doi in common_dois:
                continue
            if doi == '':
                doi = filename
            common_dois.append(doi)
            new_data[id] = journal_data[id]
            original_abstract = metadata[doi][0]
            original_title = metadata[doi][1]
            new_data[id]['year'] = str(metadata[doi][2])
            if str(d['abstract']).lower() != str(original_abstract).lower():
                new_data[id]['abstract'] = original_abstract
                logging.warning(
                    f"For DOI: {doi} , there is mismatch in abstract. \nOriginal:{original_abstract} \nFrom XML:{d['abstract']}. \n Replacing with original abstract.")
            if d['title'] != original_title:
                new_data[id]['title'] = original_title
                logging.warning(
                    f"For DOI: {d['doi']} , there is mismatch in title. \nOriginal:{original_title} \nFrom XML:{d['title']}. \n Replacing with original title.")
        elif filename in metadata.keys():
            doi = filename
            common_dois.append(doi)
            new_data[id] = journal_data[id]
            original_abstract = metadata[doi][0]
            original_title = metadata[doi][1]
            new_data[id]['year'] = str(metadata[doi][2])
            if str(d['abstract']).lower() != str(original_abstract).lower():
                new_data[id]['abstract'] = original_abstract
                logging.warning(
                    f"For DOI: {doi} , there is mismatch in abstract. \nOriginal:{original_abstract} \nFrom XML:{d['abstract']}. \n Replacing with original abstract.")
            if d['title'] != original_title:
                new_data[id]['title'] = original_title
                logging.warning(
                    f"For DOI: {d['doi']} , there is mismatch in title. \nOriginal:{original_title} \nFrom XML:{d['title']}. \n Replacing with original title.")
        else:
            stats['doi_not_in_original'].append(doi)
            logging.warning(f"DOI: {doi} does not exist.")

    stats['doi_not_in_xml'] = list(set(metadata.keys()) - set(xml_dois))
    print(stats)
    return new_data

def verify_journal(journal_data, csv_file):
    """
    This methods creates a metadata dictionary containing abstract, title and year of each article present in metadata
    file. This will be used to verify and possibly update the data saved in journal_data extracted from xmls.

    :param journal_data: dictionary containing data extracted from xmls
    :param csv_file: metadata containing information like title, abstract, authors etc for each article
    :return: updated dictionary containing information for each article.
    """
    df = pd.read_csv(csv_file)
    metadata = defaultdict()
    for doi, abs, title, year in zip(df['DOI'].values, df['Abstract'].values, df['Title'].values, df['Year'].values):
        doi = str(doi).lower()
        abs = str(abs)
        title = str(title)
        if doi != 'nan':
            metadata[doi] = (abs, title, year)
    updated_journal_data = update_data(journal_data=journal_data, metadata=metadata)
    return updated_journal_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for verify and update data extracted for scientific articles from xml')

    parser.add_argument(
        '-json_file',
        help='Path to json file',
        type=str, default='./data/jsons/EIST.json',
    )

    parser.add_argument(
        '-csv_file',
        help='Path to csv file',
        type=str, default='./data/pdfs/EIST_PDFS_TM/eist574.csv',
    )


    args, remaining_args = parser.parse_known_args()

    with open(args.json_file) as f:
        data = json.load(f)

    data = verify_journal(data, args.csv_file)

    with open(args.json_file, "w+") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)