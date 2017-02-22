import sqlite3
from HTMLParser import HTMLParser
from time import time
import pandas as pd
import itertools
import numpy as np


conn = sqlite3.connect('articles.sqlite')
cur = conn.cursor()

# select top authors
top_author = 'SELECT author_unique FROM Counts WHERE count >= 50 ORDER BY count DESC LIMIT 10'

author_doc = []
author_lst = []

for i in cur.execute(top_author) :
    author_lst.append(i[0])
#print "number of unique authors: ", len(author_lst)
# print "Authors: ", author_lst

d={}
list_authors = []
for i, author_correct in enumerate(author_lst):
    # print "author we want: ", author_correct

    cur.execute('''SELECT abstract, author FROM Articles WHERE author LIKE ? ''', ('%{}%'.format(author_correct), ) )
    all_rows = cur.fetchall()

    authors_list = [x[1] for x in all_rows]
    docs = [x[0] for x in all_rows]

    for author, row in itertools.izip(authors_list, docs) :
        # print "Fetched authors: ", author
        authors = author.split('; ')
        for a in authors:
            if author_correct == a:
                author_doc.append(row)
                list_authors.append(a)
    conn.commit()

df = pd.DataFrame({'author' : list_authors, 'doc': author_doc})
print "size: ", df.shape
print df.head()

cur.close()

# Data exploration
print df['author'].value_counts()
# print df['doc'].map(len).max()

# Pad documents to the same max length as the longest doc
padded_docs = []
vocab = []

max_length = df['doc'].map(len).max()
# padding_word = ' </PAD>'
lengths = []
for i in range(df.shape[0]):

    doc = df['doc'][i]
    doc_split = doc.split(" ")
    lengths.append(len(doc_split))

max_length = max(lengths)
print "max doc length: ", max_length      # 319

    # print "length of doc: ", len(doc_split)
for i in range(df.shape[0]):

    doc = df['doc'][i]
    doc_split = doc.split(" ")

    # print "length:" , len(doc_split)
    padding_num = max_length - len(doc_split)
    # print "padding num: ", padding_num
    doc_new = doc + " </PAD>" * padding_num
    # print doc_new
    # print "length of padded doc: ", len(doc_new)
    # # print "doc: ", doc_new
    words = doc_new.split(" ")
    # print "length of words: ", len(words)
    vocab.append(words)
    padded_docs.append(doc_new)

# print "Padded doc: ", padded_docs
# print "split words: ", vocab
from collections import Counter
word_counts = Counter(itertools.chain(*vocab))

# Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]
print "=" * 80
# print(vocabulary_inv)

# print "=" * 80
# print "vocab inv: ", vocabulary_inv
# Mapping from word to index
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# Map docs and labels (authors) to vectors based on a vocabulary
print "vocab shape: ", len(vocab)
# print "vocab[0]: ", vocab[0]

x = np.array([ [ vocabulary[word] for word in doc ] for doc in vocab ])
print "x size: ", x.shape
# # print "=" * 80
# print x

# Convert labels to factor for Tensor Flow
# labels = df['author']
labels = df['author'].astype('category')
# print "labels: ", labels
labels = labels.cat.codes
# print "=" * 80
# print "labels: ", labels

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(labels)
# print "classes: ", lb.classes_
labels = lb.transform(labels)
# print "transformed labels: ", labels

# Stratified split of data and labels
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, test_index in sss.split(x, labels):
    # print "train index: ", train_index
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

print "Train/test split: %d/%d" % (len(y_train), len(y_test))
print 'train shape:', x_train.shape
print 'test shape:', x_test.shape
print 'vocab_size', len(vocabulary)
print 'sentence max words', x_train.shape[1]
