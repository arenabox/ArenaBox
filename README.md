### ArenaBox

## Introduction

todo

## Setup

It is recommended to create a conda environment before setting up the project.
```
# Execute following command. We use python v3.9 
conda create -n ENV_NAME python=3.9

# replace ENV_NAME with a name of your choice. Once environment is created, activate the environment as:
conda activate ENV_NAME
```

Start by installing required packages mentioned in requirement.txt. If installation of BERTopic results in error related to 
[HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan), then first install HDBSCAN using following command:

```
pip install --upgrade git+https://github.com/scikit-learn-contrib/hdbscan.git#egg=hdbscan
```
and then install required packages 

```
pip install -r requirements.txt
```

Modelling can be made faster if high end resources are available. We used NVIDIA-??
with CUDA 11.6 setup. As mentioned in the docs for [BERTopic](https://maartengr.github.io/BERTopic/faq.html#can-i-use-the-gpu-to-speed-up-the-model)
we installed cuML to speedup HDBSCAN and UMAP and make full use of available resources.
## Usage

### 1. Training
In order to perform topic modelling, we use `topic_modelling.py` 
script which can model a single topic or all topics at once
```
python topic_modelling.py
# Use --topic_name parameter to model a particular topic, eg:
python topic_modelling.py --topic_name euparl

# By default, all topics will be modelled at once
```
<!--
**arenabox/ArenaBox** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
