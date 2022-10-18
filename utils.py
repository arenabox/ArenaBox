import re
import string
import nltk
from nltk.corpus import stopwords
import preprocessor as p
import en_core_web_md


def setup():
    nltk.download('stopwords')
    nlp = en_core_web_md.load()
    return nlp

def clean_text(text, nlp):

    ## Remove @ from mentions
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    #p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
    text = p.clean(text)

    # POS preprocessing
    removal = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']
    text = ' '.join([str(token) for token in nlp(text) if token.pos_ not in removal])



    '''text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)'''

    return text