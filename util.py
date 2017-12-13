import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

def tokenize_and_stem(text):
    regex = re.compile('([^\w]+|\d+)')
    tokens = nltk.word_tokenize(regex.sub(' ', text))
    output = []
    for t in tokens:
        stem = WordNetLemmatizer().lemmatize(t, pos='v')
        stem = WordNetLemmatizer().lemmatize(stem, pos='a')
        stem = WordNetLemmatizer().lemmatize(stem, pos='s')
        stem = WordNetLemmatizer().lemmatize(stem, pos='r')
        stem = WordNetLemmatizer().lemmatize(stem, pos='n')
        if len(stem) > 2:
            output.append(stem)
    return output
