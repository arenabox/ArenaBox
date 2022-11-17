import re
import string
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import preprocessor as p
import en_core_web_md #python -m spacy download en_core_web_md

class Utils():
    def __init__(self):
        nltk.download('stopwords')
        self.nlp = en_core_web_md.load()
        self.all_stopwords = []
        for lang in stopwords.fileids():
            self.all_stopwords += stopwords.words(lang)


    def preprocess_text(self, text):

        ## Remove @ from mentions
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

        ## Remove puncuation
        text = text.translate(string.punctuation)

        ## Convert words to lower case and split them
        text = text.lower().split()

        ## Remove stop words from multiple languages
        text = list(set(text) - set(text).intersection(set(self.all_stopwords)))
        #text = [w for w in text if not w in set(all_stopwords) and len(w) >= 3]

        text = " ".join(text)

        # Clean the text
        #p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
        text = p.clean(text)



        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

        return text


    def pos_preprocessing(self, docs, tags_to_remove):
        new_docs = []
        for doc in self.nlp.pipe(docs, n_process=4):
            tokens = [str(token) for token in doc if
                        token.pos_ not in tags_to_remove and not token.is_stop and token.is_alpha]
            new_docs.append(" ".join(tokens))
        return new_docs