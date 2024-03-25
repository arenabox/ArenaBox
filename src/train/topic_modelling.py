import json

from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, TextGeneration
import pickle
import numpy as np
import os

from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# System prompt describes information given to all conversations
system_prompt = """
[INST] <>
You are a helpful, respectful and honest assistant for labeling topics.
<>
"""
# Example prompt demonstrating the output we are looking for
example_prompt = """
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] Environmental impacts of eating meat
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""
prompt = system_prompt + example_prompt + main_prompt

def create_llm_pipeline():


    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")

    model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ", device_map='auto')

    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )

    return pipe
def get_data(data_path, typ= 'json'):
  if typ == 'txt':
    with open(data_path, 'r',encoding="utf-8") as f:
      data = f.read().replace('\n', '').rstrip()
    f.close()
    all_text = data.split('.')
    # TODO: integrate preprocessing pipeline here
    #all_chunks = create_chunks(all_text)
  else:
    with open(data_path, 'r') as f:
      data = json.load(f)
    f.close()
    all_text = []
    if 'text' not in data:
      for user, user_data in data.items():
        text = user_data['text']
        all_text.extend(text)
      all_text = ' '.join(all_text).split('.')
      #all_chunks = create_chunks(all_text)
    else:
      all_chunks = data['text']

  return all_chunks
class Dimensionality:
  """ Use this for pre-calculated reduced embeddings """
  def __init__(self, reduced_embeddings):
    self.reduced_embeddings = reduced_embeddings

  def fit(self, X):
    return self

  def transform(self, X):
    return self.reduced_embeddings


class TopicModelling:
  def __init__(self, base_path, config):
    self.base_path = base_path

    self.config = config
    self.embedding_model_name = self.config['train']['embedding_model']
    self.prepare_paths()

  def save_model(self):
      save_path = os.path.join(self.config['model_path'], 'final')
      self.topic_model.save(save_path, serialization="safetensors",
                            save_ctfidf=True, save_embedding_model=self.embedding_model_name)


  def load_model(self, model_name='final'):
    self.topic_model = BERTopic.load(os.path.join(self.config['model_path'], model_name))
    self.load_embeddings()
    self.load_data()
    print('Model Loaded Successfully')
    return self.topic_model

  def prepare_paths(self):
    self.data_path = os.path.join(self.base_path, 'only_text.json')
    self.embeddings_path = os.path.join(self.base_path, 'embeddings.npy')
    self.umap_embeddings_path = os.path.join(self.base_path, 'umap_embeddings.npy')
    self.umap_embeddings_path_2d = os.path.join(self.base_path, 'umap_2d_embeddings.npy')
    self.vocab_path = os.path.join(self.base_path, 'vocab.txt')

  def load_data(self):
    print('Fetching data....')
    self.data = get_data(self.data_path)

  def load_embeddings(self):
    print('Fetching Embeddings...')
    self.embeddings = np.load(self.embeddings_path)
    self.reduced_embeddings = np.load(self.umap_embeddings_path)
    self.reduced_embeddings_2d = np.load(self.umap_embeddings_path_2d)

  def load_vocab(self):
    print('Fetching vocab...')
    with open (self.vocab_path, 'rb') as fp:
        self.vocab = pickle.load(fp)

  def load_models(self):
    print('Loading Models...')
    self.embedding_model = SentenceTransformer(self.embedding_model_name)
    self.umap_model = Dimensionality(self.reduced_embeddings)
    self.hdbscan_model = BaseCluster()
    # Find clusters of semantically similar documents
    hdbscan_model = HDBSCAN(
        **self.config['train']['cluster']
    )
    self.clusters = hdbscan_model.fit(self.reduced_embeddings).labels_
    sw = stopwords.words()
    self.vectorizer_model = CountVectorizer(vocabulary=self.vocab, stop_words=sw)
    keybert_model = KeyBERTInspired()

    # Part-of-Speech
    pos_model = PartOfSpeech("en_core_web_sm")

    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    #Uncomment following line for using llama model for labelling
    pipe = create_llm_pipeline()
    llama2 = TextGeneration(pipe, prompt=prompt)

    # All representation models
    self.representation_model = {
        "KeyBERT": keybert_model,
        # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "MMR": mmr_model,
        "POS": pos_model,
         "Llama2": llama2, # Uncomment for using Llama
    }

  def get_topic_model(self):
    self.load_data()
    self.load_embeddings()
    self.load_vocab()
    self.load_models()
    print('Modelling...')
    self.topic_model= BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model,
            **self.config['train']['bertopic'],
    ).fit(self.data, embeddings=self.embeddings, y=self.clusters)

    #Uncomment for using llama labels
    llama2_labels = [label[0][0].split("\n")[0] for label in self.topic_model.get_topics(full=True)["Llama2"].values()]
    self.topic_model.set_topic_labels(llama2_labels)

    return self.topic_model