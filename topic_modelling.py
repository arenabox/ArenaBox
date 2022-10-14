import json
from os import listdir
from os.path import isfile, join

import hdbscan
import umap
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

base_jsonl_folder = 'data/eit_jsonl'
eit_json_files = [f for f in listdir(base_jsonl_folder) if isfile(join(base_jsonl_folder, f)) and f.endswith('json')]

docs = []
for eit_json_file in eit_json_files:
    community_name = eit_json_file.split('.')[0]
    with open(join(base_jsonl_folder,eit_json_file), "r") as fd:
        eit_data_json = json.load(fd)
    fd.close()
    for id, data in eit_data_json.items():
        docs.append(data['content'])


sentence_model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
embeddings = sentence_model.encode(docs, show_progress_bar=True)
umap_model = umap.UMAP(n_neighbors=15,
                       n_components=10,
                       min_dist=0.0,
                       metric='cosine',
                       low_memory=False)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10,
                                min_samples=1,
                                metric='euclidean',
                                cluster_selection_method='eom',
                                prediction_data=True)
model = BERTopic(top_n_words=20,
                       n_gram_range=(1,2),
                       calculate_probabilities=True,
                       umap_model= umap_model,
                       hdbscan_model=hdbscan_model,
                       similarity_threshold_merging=0.5,
                       verbose=True)
topics, probabilities = model.fit_transform(docs, embeddings)
model.save("combined_model")