import sqlite3
from HTMLParser import HTMLParser
import os
from time import time
import datetime
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

d={}
list_authors = []
for i, author_correct in enumerate(author_lst):
    cur.execute('''SELECT abstract, author FROM Articles WHERE author LIKE ? ''', ('%{}%'.format(author_correct), ) )
    all_rows = cur.fetchall()

    authors_list = [x[1] for x in all_rows]
    docs = [x[0] for x in all_rows]

    for author, row in itertools.izip(authors_list, docs) :
        authors = author.split('; ')
        for a in authors:
            if author_correct == a:
                author_doc.append(row)
                list_authors.append(a)
    conn.commit()

df = pd.DataFrame({'author' : list_authors, 'doc': author_doc})
cur.close()

# Data exploration
print df['author'].value_counts()

# Pad documents to the same max length as the longest doc
vocab = []
lengths = []

for i in range(df.shape[0]):
    doc = df['doc'][i]
    doc_split = doc.split(" ")
    lengths.append(len(doc_split))

max_length = max(lengths)
print "max doc length: ", max_length

for i in range(df.shape[0]):
    doc = df['doc'][i]
    doc_split = doc.split(" ")
    # How much to pad each doc
    padding_num = max_length - len(doc_split)
    doc_new = doc + " </PAD>" * padding_num
    words = doc_new.split(" ")
    vocab.append(words)


from collections import Counter
word_counts = Counter(itertools.chain(*vocab))

# Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]

# Mapping from word to index
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# Map docs and labels (authors) to vectors based on a vocabulary
print "vocab shape: ", len(vocabulary)
x = np.array([ [ vocabulary[word] for word in doc ] for doc in vocab ])
print "x size: ", x.shape

# Convert labels to factor for Tensor Flow
labels = df['author'].astype('category')
labels = labels.cat.codes
labels_unique = np.unique(labels)

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(labels)
# print "classes: ", lb.classes_
labels_tf = lb.transform(labels)
print "labels: ", labels_tf.shape
# print "transformed labels: ", labels

# Stratified split of data and labels
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, test_index in sss.split(x, labels_tf):
    # print "train index: ", train_index
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = labels_tf[train_index], labels_tf[test_index]




vocab_size = len(vocabulary)
doc_size = x_train.shape[1]
print "Train/test split: %d/%d" % (len(y_train), len(y_test))
print 'train shape:', x_train.shape
print 'test shape:', x_test.shape
print 'vocab_size', len(vocabulary)
print 'sentence max words', x_train.shape[1]
# print "vocab: ", vocabulary
# print x_train[0]
import tensorflow as tf

num_classes = len(labels_unique)

x = tf.placeholder(tf.int32, [None, x_train.shape[1]], name="input_x")

# y_ is the correct classes
y_ = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

# Keeping track of l2 regularization loss (optional)
l2_loss = tf.constant(0.0)

# embedding_size = 128
embedding_size = 500

filter_sizes = [3,4,5]
# num_filters = 128

num_filters = 300
# Embedding layer
W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedded_chars = tf.nn.embedding_lookup(W, x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Create a convolution + maxpool layer for each filter size
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    # Convolution Layer
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
    conv = tf.nn.conv2d(
        embedded_chars_expanded,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID")
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b))
    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(
        h,
        ksize=[1, doc_size - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')
    pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(3, pooled_outputs)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
keep_prob = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(h_pool_flat, keep_prob)

# Final (unnormalized) scores and predictions
W = tf.get_variable(
    "W",
    shape=[num_filters_total, num_classes],
    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
l2_loss += tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)
scores = tf.nn.xw_plus_b(h_drop, W, b)
predictions = tf.argmax(scores, 1)

# Calculate Mean cross-entropy loss
l2_reg_lambda=0.0

losses = tf.nn.softmax_cross_entropy_with_logits(scores, y_)
cross_entropy = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

# Accuracy
correct_predictions = tf.equal(predictions, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

t0 = time()
for i in range(500):
    # batch = mnist.train.next_batch(50)
    if i%100 == 0:

        train_accuracy = accuracy.eval(session=sess, feed_dict={ x: x_train, y_: y_train, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: x_train, y_: y_train, keep_prob: 0.5})


tt = time()-t0
print("Training TensorFlow NN took: {}").format(round(tt,3))

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: x_test, y_: y_test, keep_prob: 1.0}))
