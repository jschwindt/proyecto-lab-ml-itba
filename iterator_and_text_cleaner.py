import os
import re
import numpy as np

# Sample Usage:
#
#    sentence_iterator = NewsIterator('./sentences', 100)
#
#    for x in sentence_iterator:
#        print(x)
#

def non_alpha_cleaner(orig_text):
    return re.sub(r'[\W_]+', ' ', orig_text)

class NewsIterator(object):
    def __init__(self, dirname, max_words = 100, padding_size = 5):
        self.dirname = dirname
        self.max_words = max_words
        self.padding_size = padding_size

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            with open(self.dirname + fname) as f:
                text = non_alpha_cleaner(f.read()).lower()
                words = text.split()[0:self.max_words-self.padding_size]
                for i in range(len(words), self.max_words):
                    words.append('<PAD>')
                if len(words) < 800:
                    yield words

class SimpleIterator(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            with open(self.dirname + fname) as f:
                words = non_alpha_cleaner(f.read()).lower().split()
                if len(words) < 800:
                    yield words

class FileWordsIterator(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            with open(self.dirname + fname) as f:
                words = non_alpha_cleaner(f.read()).lower().split()
                if len(words) < 800:
                    yield fname, words

def article_matrix(w2v_model, filename, shape):
    word_matrix = np.zeros(shape)
    num_words, vec_size = shape
    i = 0
    with open(filename) as f:
        words = non_alpha_cleaner(f.read()).lower().split()
        for word in words:
            if word in w2v_model.wv:
                word_matrix[i] = w2v_model.wv.word_vec(word)
                i += 1
                if i == num_words:
                    break
    return word_matrix

def article_indices(word2index, filename, num_words):
    i = 0
    windices = np.zeros(num_words, dtype=np.int)
    with open(filename) as f:
        words = non_alpha_cleaner(f.read()).lower().split()
        for word in words:
            if word in word2index:
                windices[i] = word2index[word]
                i += 1
                if i == num_words:
                    break
    return windices

def text_indices(word2index, text, num_words):
    i = 0
    windices = np.zeros(num_words, dtype=np.int)
    words = non_alpha_cleaner(text).lower().split()
    for word in words:
        if word in word2index:
            windices[i] = word2index[word]
            i += 1
            if i == num_words:
                break
    return windices

class File2MatrixIterator(object):
    def __init__(self, file_list, w2v_model, shape):
        self.file_list = file_list
        self.model = w2v_model
        self.shape = shape

    def __iter__(self):
        for fname in self.file_list:
            yield article_matrix(self.model, fname, self.shape)
