# METAFRASIA

This study envisions a metatheoretical framework and research agenda for socio-institutional analysis(SIA) in sustainability transitions research by performing a computational literature review and social x-ray of three distinct academic discourse networks.By relating Research in Sociology of Organizations, Environmental Innovation, and Sustainability Science and by employing advanced machine learning techniques, including Latent Dirichlet Allocation (LDA) and BERTopic, we analyze a vast corpus of 2,360 articles, totaling approximately 9.8 million words. Our hybrid socio-semantic approach involves (a) an interstitial mapping of authors in these fields, utilizing a 'social x-ray' method, and (b) a computational literature review to discern 138 distinct research themes. This endeavor not only catalogues the focal topics across these journals but also highlights the necessity for more nuanced, interstitial research within this linguistic topography. Consequently, we are envisioning an interstitial research agenda, comprising five research directions and a 15-point action plan aimed at fostering interdisciplinary dialogue and advancing theoretical sociology driven sustainability transitions research in the emerging domain of SIA . Our methodological innovation demonstrates the value of computational analytics in conceptualizing and navigating the complex networks of socio-institutional discourse, setting a new precedent for future interdisciplinary research.


## Usage


<details open>
<summary>Data Collection </summary>

For different data sources, we collect data differently.  
<details open>
<summary>Scientific Articles  </summary>

We collect pdfs for the articles we want to process, then we convert pdfs to xml 
using [pdf_to_xml.sh](src/data/collection/pdf_to_xml.sh) script. More information can be found inside the 
script. Example usage:

````shell
grobid_client --input "PATH/TO/PDF/FILES/" --output "PATH/TO/SAVE/XML/FILES/" processFulltextDocument
````
</details>

<details open>
<summary>Tweets </summary>

In order to collect tweets we use [Snscrape](https://github.com/JustAnotherArchivist/snscrape) python client. We 
use [tweets.py](src/data/collection/tweets.py) script to retrieve tweets. This script 
provide following functionality  
* Extract tweets from list of user
* Extract tweets where list of users are mentioned
* Extract tweets for the mentioned users in the tweets of given user

We define a [config](configs/data/collection/tweet.yaml) that can be used to select the kind of tweet dataset we
want to crawl. Example config:

````yaml
save_folder: ./data/collection/tweets # folder to save tweets
users: # list of users for tweet crawl
  - EITHealth
  - EU_Commission

from_users: True # if enabled then tweets for given list of users will be crawled
have_user_mentioned: False # if enabled then tweets which mentions the user will be crawled

from_mentioned_users: False # If true then looks for tweet jsons in save_folder for all the users and crawl tweets
  # for mentioned users in the tweets.
mentioned_by_params:
  top_n_users: 400 # Get tweets for the top_n_users mentioned by a particular user. It is used when mentioned_by_user is enabled
  interaction_level: 3 # Save those mentioned users who have mentioned original user interaction_level times

get_images: False # extract images from tweets of users
# Optional, comment them out if not required. All the tweets will be mined in that case.
since: 2022-10-01
until: 2022-10-30

````

Example usage:

````shell
python src/data/collection/tweets.py -config configs/data/collection/tweet.yaml
````

For above-mentioned example config, tweets for two users (EITHealth and EU_Commission) will be crawled 
between the 2022-10-01 to 2022-10-30.

</details>

<details open>
<summary>Website</summary>

todo ...
</details>

</details>

<details open>
<summary >
Data Extraction
</summary>

Data collected for tweets using snscrape are in format ready for preprocessing, 
so we do not need extraction process for twitter data.  
For scientific articles, we get files in `tei.xml` format after converting it from
pdf in previous step. We use [sci_articles.py](src/data/extraction/sci_articles.py)
script to  extract required information from xml. We list information
that we need from scientific article in [config file](configs/data/extraction/sci_articles.yaml).
Example config:

````yaml
type: sci
corpus_name: EIST # this will be used to save processed file as {corpus_name}.json
info_to_extract: # list of information to extract from xml file
  - doi
  - title
  - abstract
  - text
  - location
  - year
  - authors
````
Following command is used :

````shell
python src/data/extraction/sci_articles.py -config configs/data/extraction/sci_articles.yaml
````
A json file will be created which will contain extracted information
for all the xml files. 

````json
{
    "1": {
        "file_name": "-It-s-not-talked-about---The-risk-of-failure-_2020_Environmental-Innovation-",
        "doi": "10.1016/j.eist.2020.02.008",
        "title": "Environmental Innovation and Societal Transitions",
        "abstract": "Scholars of sustainability transition have given much attention to local experiments in 'protected spaces' where system innovations can be initiated and where learning about those innovations can occur. However, local project participants' conceptions of success are often different to those of transition scholars; where scholars see a successful learning experience, participants may see a project which has failed to \"deliver\". This research looks at two UK case studies of energy retrofit projects -Birmingham Energy Savers and Warm Up North, both in the UK, and the opportunities they had for learning. The findings suggest that perceptions of failure and external real world factors reducing the capacity to experiment, meant that opportunities for learning were not well capitalised upon. This research makes a contribution to the sustainability transitions literature which has been criticised for focusing predominantly on successful innovation, and not on the impact of failure.",
        "text": {
            "Introduction": " A transition away from the use of fossil fuels to heat and power inefficient homes is urgent if the worst climate change predictions are to be avoided. Such a transition is complex, long term and uncertain, requiring change in multiple subsystems which are locked-in to high carbon usage (Unruh, 2000). Scholars of tran ..."
````

We go through an additional step to verify the data extracted for scientific
articles from xml by validating it using a metatdata(csv) for articles generated
using [Scopus](https://www.scopus.com). We use following command to perform this task

```shell
python src/utils/verify_sci_articles.py -json_file EIST.json -csv_file EIST_metadata.csv
```

This validation step can also be done along with extraction if we add
`metadata_csv` key to the [data extraction config](configs/data/extraction/sci_articles.yaml).

</details>


<details open>
<summary>Data Preprocessing</summary>

<details open>
<summary>Twitter Data</summary>

Preprocessing for tweets includes removing mentions, stopwords, punctuations
and cleaning tweets using [tweet-preprocessor](https://github.com/s/preprocessor).
We provide preprocessing [config](configs/data/preprocess/tweets.yaml) to choose from these functionalities. Example 
config:

````yaml
type: tweets
content_field: renderedContent # name of the key containing tweet content, update if it is something else in mined tweet data
preprocess:
  remove_mentions: True
````

Preprocessing can be done using following command:

````shell
python src/data/preprocess/tweets.py -user EITHealth
````
This will save preprocessed tweets for user in a json file.  

**Note**: You can skip `user` parameter if you want to preprocess tweets for 
all users at once.
</details>

<details open>
<summary>Scientific Articles</summary>

Preprocessing for scientific articles is similar to tweets which includes
removing stopwords, punctuations, etc. We provide preprocessing 
[config](configs/data/preprocess/sci_articles.yaml) to choose from these 
functionalities. Example config:

````yaml
type: sci
preprocess:
  remove_paran_content: True # to remove reference from the text
  noisywords: # give a list of noisy words to remove
      - et
      - b.v.
      - ©
      - emerald
      - elsevier
  remove_pos: # pos tags from spacy: https://spacy.io/usage/linguistic-features
    - ADV
    - PRON
chunk: # used to break long sentences into small chunks within range (min_seq_len, max_seq_len)
  max_seq_len: 512
  min_seq_len: 100
````

We provide an option to divide large text into small text using `chunk` key.
Preprocessing can be done using following command:

````shell
python src/data/preprocess/sci_articles.py -journal EIST
````
This will save preprocessed articles for a particular journal in a json file.  

**Note**: You can skip `journal` parameter if you want to preprocess articles for 
all journals at once.

</details>

</details>

<details open>
<summary>Training</summary>

We use [Bertopic](https://github.com/MaartenGr/BERTopic) for topic modelling. Our
pipeline requires data in following format:

````json
{
  "text": [
    "list of text to be used for topic modelling"
  ]
}
````

Data should be in json format and it must contain a key `text` which will be used
for topic modelling. Value of this key is list of string which represent document.
Data json can have other keys as well for example we have `title`, `id`, `class` etc.
as metadata for each document in `text`.

To run topic modelling we use [train config](configs/train/base.yaml), snippet
of config is shown below:

````yaml
type: sci
model_name: EIST
train:
  supervised: False
  n_gram_range: # Give a range here. In example here, we use ngram from 1-3
    - 1
    - 3
  embedding_model:
    use: sentence_transformer
    combine: False
    name: paraphrase-multilingual-mpnet-base-v2 # allenai/scibert_scivocab_uncased
    max_seq_len: 512

````

This config is used along with [topic_model.py](src/train/topic_model.py) script to 
perform topic modelling.

````shell
python src/train/topic_model.py -config configs/train/base.yaml
````

</details>


<details open>
<summary>Evaluation</summary>

Currently, evaluation is only performed to get optimal number of topics for given
dataset. It can be done independent of training pipeline,

````shell
python src/evaluate/using_lda.py -config configs/evaluate/lda.yaml
````

or can be integrated to train pipeline by adding `find_optimal_topic_first` key
to [train config](configs/train/base.yaml) as :

````yaml
...
find_optimal_topic_first:
  config_path: ./configs/evaluate/lda.yaml
...
````


</details>


<details open>
<summary>Insights and Visualization</summary>

**Note**: This section is only validated against scientific articles pipeline yet.

There are several insights that can be made from the topics. Topic loading can be used to understand temporal
trends like hot topics, cold topics, evergreen topics etc in the data. We can obtain this using the following 
command:

```shell
python src/analyze/temporal_trends.py
```

Successful execution of this command will create multiple files,

```
csvs
  sci
    topic_landscape.csv
    descriptive_stats.csv
    temporal_landscape.csv
plots
  sci
    loading_heatmap.jpg
    hot_topics.jpg
    cold_topics.jpg
    ...
```


</details>


## Tools Used
1. [Grobid](https://github.com/kermitt2/grobid_client_python)
2. [Snscrape](https://github.com/JustAnotherArchivist/snscrape)
3. Hyphe
4. Bertopic
5. ...
