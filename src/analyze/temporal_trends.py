from os.path import join
import json
from collections import defaultdict, Counter, OrderedDict
from os.path import join
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
import yaml
from bertopic import BERTopic
from sklearn.metrics import r2_score


def plot_loading_heatmap(topic_loading, config, plot_name):
    fig, ax = plt.subplots(figsize=(20, 15))
    figheat = sns.heatmap(topic_loading, ax=ax)
    figheat.set(xlabel=config['x_label'], ylabel=config['y_label'])
    fig.savefig(plot_name)
def get_topic_loadings(topic_doc, doc_id2idx, total_docs, config, plot_path=None):
    # get loading data
    loading_data = defaultdict(list)
    for i in range(0, total_docs):
        probs = list(topic_doc.loc[doc_id2idx[i], :].mean())
        probs = [j / sum(probs) for j in probs]
        loading_data[i] = probs
    ldf = pd.DataFrame(loading_data)

    if 'topic_loading' in config:
        plot_name = join(plot_path, f'{config["name"]}_loading_heatmap.jpg')
        plot_loading_heatmap(ldf, config['topic_loading'], plot_name)

    return ldf

def fetch_metadata(metadata_path, metadata_dict, name):
    og_req_data = defaultdict(dict)
    if name == 'all':
        for k, v in metadata_dict.items():
            og_metadata = pd.read_csv(join(metadata_path, v))
            for index, row in og_metadata.iterrows():
                og_req_data[row['Title']] = {'Authors': row['Authors'], 'Volume': row['Volume']}
    else:
        og_metadata = pd.read_csv(join(metadata_path, metadata_dict[name]))
        for index, row in og_metadata.iterrows():
            og_req_data[row['Title']] = {'Authors': row['Authors'], 'Volume': row['Volume']}

    return og_req_data

def prepare_full_metadata(data, doc_id2idx, total_docs, metadata):
    doc_id2year = defaultdict(int)
    doc_id2title = defaultdict(int)
    doc_id2volume = defaultdict(int)
    doc_id2authors = defaultdict(str)
    for i in range(0, total_docs):
        idxs = doc_id2idx[i]
        if len(idxs)<1:
            continue
        title = data.loc[doc_id2idx[i][0], 'title']
        doc_id2year[i] = data.loc[doc_id2idx[i][0], 'time']
        doc_id2title[i] = title
        doc_id2authors[i] = metadata[title]['Authors']
        doc_id2volume[i] = metadata[title]['Volume']

    full_metadata = {'title':doc_id2title, 'authors':doc_id2authors, 'year':doc_id2year,
                     'volume':doc_id2volume}

    return full_metadata

def get_topic_landscape(topic_loading, topic2words, full_metadata, config, csv_path):
    topic_landscape = defaultdict(list)
    topic_landscape['Topic Label'] = topic_names
    for i in range(0, len(topic2words)):
        doc_id = topic_loading.T[i].idxmax()
        topic_landscape['topic_nr'].append(i)
        topic_landscape['most_freq_words'].append(topic2words[i])
        topic_landscape['rep_doc_year'].append(full_metadata['year'][doc_id])
        topic_landscape['title'].append(full_metadata['title'][doc_id])
        topic_landscape['volume'].append(full_metadata['volume'][doc_id])
        topic_landscape['authors'].append(full_metadata['authors'][doc_id])
    if config['topic_landscape']:
        file_name = join(csv_path, f'{config["name"]}_topic_landscape.csv')
        pd.DataFrame(topic_landscape).to_csv(file_name)
    return topic_landscape

def get_descriptive_stats(topic_loading, config, csv_path):
    loading_means = topic_loading.mean(axis=1) / topic_loading.mean(axis=1).mean()
    loading_maxs = topic_loading.max(axis=1)
    loading_mins = topic_loading.min(axis=1)
    descriptive_stats = defaultdict(list)
    descriptive_stats['Topic Label'] = list(topic_names)
    descriptive_stats['standardized_mean'] = np.around(loading_means, 3)
    descriptive_stats['max'] = np.around(loading_maxs, 3)
    descriptive_stats['min'] = np.around(loading_mins, 3)

    if config['descriptive_stats']:
        file_name = join(csv_path,f'{config["name"]}_descriptive_stats.csv')
        pd.DataFrame(descriptive_stats).to_csv(file_name)

    return descriptive_stats

def get_temporal_landscape(topics2temporal_stats, total_topics, topic_names):
    count = []
    year_mean = []
    year_std = []
    year_min = []
    year_max = []
    stat_ov = defaultdict(list)
    stat_ov['topic_label'] = list(topic_names)
    for i in range(0, total_topics):
        years = list(OrderedDict(sorted(topics2temporal_stats[i].items())).keys())
        year_count = list(OrderedDict(sorted(topics2temporal_stats[i].items())).values())
        count.append(sum(year_count))
        year_mean.append(np.mean(years))
        year_std.append(np.std(years))
        year_min.append(np.min(years))
        year_max.append(np.max(years))

        stat_ov['count'].append(sum(year_count))
        # count.append(sum(y))
        model = np.poly1d(np.polyfit(years, year_count, 2))
        dft = pd.DataFrame({'x': years, 'y': year_count})
        results = smf.ols(formula='y ~ model(x)', data=dft).fit()
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, year_count)
        stat_ov['coeff_l'].append(slope)
        # coeff_l.append(slope)
        stat_ov['r_sq_l'].append(r_value * r_value)
        # r_sq_l.append(r_value * r_value)
        stat_ov['coeff_q'].append(model[2])
        # coeff_q.append(model[2])
        stat_ov['r_sq_q'].append(r2_score(year_count, model(years)))
        # r_sq_q.append(r2_score(y, model(x)))
        stat_ov['p_l'].append(p_value)
        # p_l.append(p_value)
        stat_ov['p_q'].append(results.pvalues[1])
        # p_q.append(results.pvalues[1])
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, model(years))
        # coeff_lq.append(slope)
        stat_ov['coeff_lq'].append(slope)
        # r_sq_lq.append(r_value * r_value)
        stat_ov['r_sq_lq'].append(r_value * r_value)
        # p_lq.append(p_value)
        stat_ov['p_lq'].append(p_value)

    temporal_landscape = {'topic_label': list(topic_names), 'count': count, 'year_mean': np.around(year_mean, 3),
                          'year_std': np.around(year_std, 3), 'year_min': np.around(year_min, 3),
                          'year_max': np.around(year_max, 3), 'coeff_linear': np.around(stat_ov['coeff_l'], 4),
                          'coeff_quadratic': np.around(stat_ov['coeff_q'], 4),
                          'coeff_linear_of_quadratic': np.around(stat_ov['coeff_lq'], 4)}

    return temporal_landscape, stat_ov
def plot_temporal_trajectory(topics2temporal_stats, trajac_type, used_topics, stat_ov, plot_path,config):
    top_n = config['temporal']['top_n_trend']
    plot_name = join(plot_path,f'{config["name"]}_{trajac_type}.html')
    if trajac_type == 'hot':
        curr_topics = stat_ov[(stat_ov.p_l < 0.1) & (stat_ov.coeff_l > 0)]
        curr_topics_sorted = curr_topics.sort_values(by=['p_l'])
    elif trajac_type == 'cold':
        curr_topics = stat_ov[(stat_ov.p_l < 0.1) & (stat_ov.coeff_l < 0)]
        curr_topics_sorted = curr_topics.sort_values(by=['p_l'])
    elif trajac_type == 'evergreen':
        thres = stat_ov['count'].median()
        curr_topics = stat_ov[
            (stat_ov.p_l > 0.1) & (stat_ov.p_q > 0.1) & (stat_ov['count'] >= thres)]
        curr_topics = curr_topics[~curr_topics.index.isin(used_topics)]
        curr_topics_sorted = curr_topics.sort_values(by=['count'])
    elif trajac_type == 'revival':
        curr_topics = stat_ov[(stat_ov.p_q < 0.1) & (stat_ov.coeff_q > 0)]
        curr_topics = curr_topics[~curr_topics.index.isin(used_topics)]
        curr_topics_sorted = curr_topics.sort_values(by=['p_q'])
    elif trajac_type == 'wallflower':
        thres = stat_ov['count'].median()
        curr_topics = stat_ov[
            (stat_ov.p_l > 0.1) & (stat_ov.p_q > 0.1) & (stat_ov['count'] < thres)]
        curr_topics = curr_topics[~curr_topics.index.isin(used_topics)]
        curr_topics_sorted = curr_topics.sort_values(by=['count'])
    else:
        raise KeyError(f'Given trajectory type {trajac_type}  is invalid')
    used_topics.extend(list(curr_topics_sorted.index[:top_n]))
    plot_trends(topics2temporal_stats, curr_topics_sorted.index[:top_n], name=plot_name)
    return used_topics


def plot_trends(topics2temporal_stats, indices,name):
    X = []
    Y = []
    clss = []
    years = set([b for a in [list(v.keys()) for k,v in topics2temporal_stats.items()] for b in a])
    for i in list(indices):
            x = list(OrderedDict(sorted(topics2temporal_stats[i].items())).keys())
            y = list(OrderedDict(sorted(topics2temporal_stats[i].items())).values())
            model = np.poly1d(np.polyfit(x,y, 2))

            # polynomial line visualization
            polyline = np.linspace(2011, 2022, 100)
            X.extend(polyline)
            Y.extend(model(polyline))
            clss.extend([f'{topic_names[i+1]}']*100)

    plt_df = pd.DataFrame({'Year':X, 'Frequency':Y, 'Class':clss})
    fig = px.line(plt_df, x='Year', y='Frequency', color='Class', )
    fig.write_html(name, auto_open=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for analyzing topics')

    parser.add_argument(
        '-config',
        help='Path to config file',
        type=str, default='./configs/analyse/base.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    with open(args.config) as file:
        config = yaml.safe_load(file)

    data_type = config['type']
    data_path = join(base_path['data']['preprocess'], data_type, f"{config['name']}.json")


    metadata = fetch_metadata(
    join(base_path['data']['collection'], data_type, 'metadata'), base_path[config['type']]['metadata'], config['name'])

    plot_path = join(base_path['data']['analysis']['plots'],data_type)
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    csv_path = join(base_path['data']['analysis']['csvs'], data_type)
    Path(csv_path).mkdir(parents=True, exist_ok=True)

    model_path = join('models', data_type, f"{config['name']}_topic_model", "model")
    model = BERTopic.load(model_path)
    with open(data_path, "r") as fd:
        model_data = json.load(fd)
    if 'reduce' in config:
        model.reduce_topics(model_data['text'], config['reduce'][config['name']])

    topics = model.get_topics()
    del topics[-1] # remove words which are not categorized to any topic
    topic_names = model.get_topic_info()[model.get_topic_info()['Topic']!=-1]['Name']

    topic2words = {t: [word[0] for word in words] for t, words in topics.items()}
    topic_words = [word[0] for t, words in topics.items() for word in words]
    total_docs = max(model_data['id']) + 1
    total_topics = len(topic2words)

    data = pd.DataFrame(model_data) # Dataframe containing model data
    topic_doc = pd.DataFrame(model.probabilities_) # Dataframe containing topic and their respective probability against each document

    doc_id2idx = defaultdict(list)  # mapping from document id to their respective indices in model_data or data
    if config['name'] == 'all':
        title2docs = data.groupby('title').groups
        total_docs = len(title2docs)
        for doc_id, (title, idxs) in enumerate(title2docs.items()):
            doc_id2idx[doc_id] = idxs
    else:
        for doc_id in range(0, total_docs):
            idxs = data[data.id==doc_id].index
            doc_id2idx[doc_id] = idxs

    topic_loadings = get_topic_loadings(topic_doc, doc_id2idx, total_docs, config, plot_path)

    full_metadata = prepare_full_metadata(data, doc_id2idx, total_docs, metadata)

    topic_landscape = get_topic_landscape(topic_loadings, topic2words, full_metadata, config, csv_path)

    descriptive_stats = get_descriptive_stats(topic_loadings, config, csv_path)

    if 'temporal' in config:
        representative_docs = []
        topic2docs = defaultdict(list)  # above 10% loading
        topics2temporal_stats = defaultdict(dict)
        limit = config['temporal']['threshold']
        for i in range(0, total_topics):
            ind = list(topic_loadings.T[topic_loadings.loc[i, :] > limit][i].index)
            ind = [j for j in ind if j<len(full_metadata['title'])]
            topic2docs[i] = ind
            topics2temporal_stats[i] = dict(Counter(np.array(list(full_metadata['year'].values()))[ind]))

        temporal_landscape, stat_ov = get_temporal_landscape(topics2temporal_stats, total_topics, topic_names)
        stat_ov = pd.DataFrame(stat_ov)
        temporal_landscape = pd.DataFrame(temporal_landscape)
        file_name = join(csv_path, f'{config["name"]}_temporal_landscape.csv')
        pd.DataFrame(temporal_landscape).to_csv(file_name)
        used_topics = []
        for trajac_type in config['temporal']['type']:
            used_topics.extend(plot_temporal_trajectory(topics2temporal_stats, trajac_type, used_topics, stat_ov, plot_path, config))