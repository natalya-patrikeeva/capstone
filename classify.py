import sqlite3
from HTMLParser import HTMLParser
from time import time
import pandas as pd
import itertools

conn = sqlite3.connect('articles.sqlite')
cur = conn.cursor()

# select top authors
top_author = 'SELECT author_unique FROM Counts WHERE count >= 100 ORDER BY count DESC'

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
#print "size: ", df.shape
#print df.head()

cur.close()

# Data exploration
print df['author'].value_counts()

# Extract numerical features from text content in abstracts

# Tokenizing abstract text
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words="english", analyzer = "word", ngram_range=(1, 5), token_pattern=r'\b\w+\b', max_features=10000)

data = count_vect.fit_transform(df['doc'])
#print "data: ", data.shape

# Split into train and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    data, df['author'], test_size = 0.1, random_state=3)

# print "labels_train: ", len(labels_train) # 139
# print "features_train: ", features_train.shape # (139, 5000)
# print "features_test: ", features_test.shape # (16, 5000)
###################################
# Convert labels to factor for Tensor Flow
labels_train_tf = labels_train.astype('category')
labels_train_tf = labels_train_tf.cat.codes

labels_test_tf = labels_test.astype('category')
labels_test_tf = labels_test_tf.cat.codes

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(labels_train_tf)
# print "classes: ", lb.classes_
labels_train_tf = lb.transform(labels_train_tf)
# print "transformed labels: ", labels_train_tf

lb_test = preprocessing.LabelBinarizer()
lb_test.fit(labels_test_tf)
labels_test_tf = lb_test.transform(labels_test_tf)
###################################

# Term Frequency times Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(features_train).toarray()
X_test_tfidf = tfidf_transformer.fit_transform(features_test).toarray()

# Logistic Regression - Benchmark
from sklearn import linear_model
print "starting log reg modeling..."


t0 = time()
clf = linear_model.LogisticRegression(solver='sag', max_iter=1000, random_state=42,
                                 multi_class="ovr").fit(X_train_tfidf, labels_train)
pred = clf.predict(X_test_tfidf)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

tt = time()-t0
print("Training Log Regression took: {}").format(round(tt,3))
print "Accuracy score on test data is {}.".format(round(acc,4))   

# Tensor Flow
#
# 5000 features
# 62 classes (authors)
# Naive model first
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 10000])
W = tf.Variable(tf.zeros([10000, 7]))
b = tf.Variable(tf.zeros([7]))
# model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# train
# y_ is the correct classes
y_ = tf.placeholder(tf.float32, [None, 7])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  sess.run(train_step, feed_dict={x: X_train_tfidf, y_: labels_train_tf})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# accuracy on test data.
print "Tensor Flow accuracy on test data:"
print(sess.run(accuracy, feed_dict={x: X_test_tfidf, y_: labels_test_tf}))

# Multilayer convolutional network
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([1, 10, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,1,10000,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print "h conv 1 size: ", h_conv1.get_shape()    # (?, 1, 10000, 32)

h_pool1 = max_pool_2x2(h_conv1)
print "h pool 1 size: ", h_pool1.get_shape()   # (?, 1, 5000, 32)

W_conv2 = weight_variable([1, 10, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print "h conv 2 size: ", h_conv2.get_shape()    # (?, 1, 5000, 64)
h_pool2 = max_pool_2x2(h_conv2)
print "h pool 2 size: ", h_pool2.get_shape()    # (?, 1, 2500, 64)

W_fc1 = weight_variable([1 * 2500 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 1*2500*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print "h fc1 size: ", h_fc1.get_shape()         # (?, 1024)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 7])
b_fc2 = bias_variable([7])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


t0 = time()
for i in range(2000):
  # batch = mnist.train.next_batch(50)
  if i%100 == 0:

      train_accuracy = accuracy.eval(session=sess, feed_dict={ x: X_train_tfidf, y_: labels_train_tf, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: X_train_tfidf, y_: labels_train_tf, keep_prob: 0.5})


tt = time()-t0
print("Training TensorFlow NN took: {}").format(round(tt,3))

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: X_test_tfidf, y_: labels_test_tf, keep_prob: 1.0}))
