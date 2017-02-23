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
embedding_size = 300

filter_sizes = [3,4,5]
num_filters = 128
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

# class TextCNN(object):
#     """
#     A CNN for text classification.
#     Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
#     """
#     def __init__(
#       self, sequence_length, num_classes, vocab_size,
#       embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
#
#         # Placeholders for input, output and dropout
#         self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
#         self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
#         self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
#
#         # Keeping track of l2 regularization loss (optional)
#         l2_loss = tf.constant(0.0)
#
#         # Embedding layer
#         with tf.device('/cpu:0'), tf.name_scope("embedding"):
#             W = tf.Variable(
#                 tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
#                 name="W")
#             self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
#             self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
#
#         # Create a convolution + maxpool layer for each filter size
#         pooled_outputs = []
#         for i, filter_size in enumerate(filter_sizes):
#             with tf.name_scope("conv-maxpool-%s" % filter_size):
#                 # Convolution Layer
#                 filter_shape = [filter_size, embedding_size, 1, num_filters]
#                 W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
#                 b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
#                 conv = tf.nn.conv2d(
#                     self.embedded_chars_expanded,
#                     W,
#                     strides=[1, 1, 1, 1],
#                     padding="VALID",
#                     name="conv")
#                 # Apply nonlinearity
#                 h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#                 # Maxpooling over the outputs
#                 pooled = tf.nn.max_pool(
#                     h,
#                     ksize=[1, sequence_length - filter_size + 1, 1, 1],
#                     strides=[1, 1, 1, 1],
#                     padding='VALID',
#                     name="pool")
#                 pooled_outputs.append(pooled)
#
#         # Combine all the pooled features
#         num_filters_total = num_filters * len(filter_sizes)
#         self.h_pool = tf.concat(3, pooled_outputs)
#         self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
#
#         # Add dropout
#         with tf.name_scope("dropout"):
#             self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
#
#         # Final (unnormalized) scores and predictions
#         with tf.name_scope("output"):
#             W = tf.get_variable(
#                 "W",
#                 shape=[num_filters_total, num_classes],
#                 initializer=tf.contrib.layers.xavier_initializer())
#             b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
#             l2_loss += tf.nn.l2_loss(W)
#             l2_loss += tf.nn.l2_loss(b)
#             self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
#             self.predictions = tf.argmax(self.scores, 1, name="predictions")
#
#         # CalculateMean cross-entropy loss
#         with tf.name_scope("loss"):
#             losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
#             self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
#
#         # Accuracy
#         with tf.name_scope("accuracy"):
#             correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
#             self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# def batch_iter(data, batch_size, num_epochs, shuffle=True):
#     """
#     Generates a batch iterator for a dataset.
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
#     for epoch in range(num_epochs):
#         # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]
################################################################
# Train CNN model
################################################################
# Model Hyperparameters
# tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
#
# # Training parameters
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# # Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
#
# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto(
#       allow_soft_placement=FLAGS.allow_soft_placement,
#       log_device_placement=FLAGS.log_device_placement)
#     sess = tf.Session(config=session_conf)
#     with sess.as_default():
#         cnn = TextCNN(
#             sequence_length=x_train.shape[1],
#             num_classes=len(labels_unique),
#             vocab_size=len(vocabulary),
#             embedding_size=FLAGS.embedding_dim,
#             filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
#             num_filters=FLAGS.num_filters,
#             l2_reg_lambda=FLAGS.l2_reg_lambda)
#
#         # Define Training procedure
#         global_step = tf.Variable(0, name="global_step", trainable=False)
#         optimizer = tf.train.AdamOptimizer(1e-3)
#         grads_and_vars = optimizer.compute_gradients(cnn.loss)
#         train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
#
#         # Keep track of gradient values and sparsity (optional)
#         grad_summaries = []
#         for g, v in grads_and_vars:
#             if g is not None:
#                 grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
#                 sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
#                 grad_summaries.append(grad_hist_summary)
#                 grad_summaries.append(sparsity_summary)
#         grad_summaries_merged = tf.summary.merge(grad_summaries)
#
#         # Output directory for models and summaries
#         timestamp = str(int(time.time()))
#         out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
#         print("Writing to {}\n".format(out_dir))
#
#         # Summaries for loss and accuracy
#         loss_summary = tf.summary.scalar("loss", cnn.loss)
#         acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
#
#         # Train Summaries
#         train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
#         train_summary_dir = os.path.join(out_dir, "summaries", "train")
#         train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
#
#         # Dev summaries
#         dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
#         dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#         dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
#
#         # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
#         checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
#         checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#         if not os.path.exists(checkpoint_dir):
#             os.makedirs(checkpoint_dir)
#         saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
#
#         # Write vocabulary
#         # vocab_processor.save(os.path.join(out_dir, "vocab"))
#
#         # Initialize all variables
#         sess.run(tf.global_variables_initializer())
#
#         def train_step(x_batch, y_batch):
#             """
#             A single training step
#             """
#             feed_dict = {
#               cnn.input_x: x_batch,
#               cnn.input_y: y_batch,
#               cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
#             }
#             _, step, summaries, loss, accuracy = sess.run(
#                 [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
#                 feed_dict)
#             time_str = datetime.datetime.now().isoformat()
#             print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#             train_summary_writer.add_summary(summaries, step)
#
#         def dev_step(x_batch, y_batch, writer=None):
#             """
#             Evaluates model on a test set
#             """
#             feed_dict = {
#               cnn.input_x: x_batch,
#               cnn.input_y: y_batch,
#               cnn.dropout_keep_prob: 1.0
#             }
#             step, summaries, loss, accuracy = sess.run(
#                 [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
#                 feed_dict)
#             time_str = datetime.datetime.now().isoformat()
#             print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#             if writer:
#                 writer.add_summary(summaries, step)
#
#         # Generate batches
#         batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
#         # Training loop. For each batch...
#         for batch in batches:
#             x_batch, y_batch = zip(*batch)
#             train_step(x_batch, y_batch)
#             current_step = tf.train.global_step(sess, global_step)
#             if current_step % FLAGS.evaluate_every == 0:
#                 print("\nEvaluation:")
#                 dev_step(x_dev, y_dev, writer=dev_summary_writer)
#                 print("")
#             if current_step % FLAGS.checkpoint_every == 0:
#                 path = saver.save(sess, checkpoint_prefix, global_step=current_step)
#                 print("Saved model checkpoint to {}\n".format(path))
#
#
#
#

################################################################
# CNN model
################################################################
# import mxnet as mx
# import sys,os
#
# '''
# Define batch size and the place holders for network inputs and outputs
# '''
#
# batch_size = 50 # the size of batches to train network with
# print 'batch size', batch_size
#
# input_x = mx.sym.Variable('data') # placeholder for input data
# input_y = mx.sym.Variable('softmax_label') # placeholder for output label
#
#
# '''
# Define the first network layer (embedding)
# '''
#
# # create embedding layer to learn representation of words in a lower dimensional subspace (much like word2vec)
# num_embed = 300 # dimensions to embed words into
# print 'embedding dimensions', num_embed
#
# embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
#
# # reshape embedded data for next layer
# conv_input = mx.sym.Reshape(data=embed_layer, target_shape=(batch_size, 1, doc_size, num_embed))
#
# # create convolution + (max) pooling layer for each filter operation
# filter_list=[3, 4, 5] # the size of filters to use
# print 'convolution filters', filter_list
#
# num_filter=100
# pooled_outputs = []
# for i, filter_size in enumerate(filter_list):
#     convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
#     relui = mx.sym.Activation(data=convi, act_type='relu')
#     pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(doc_size - filter_size + 1, 1), stride=(1,1))
#     pooled_outputs.append(pooli)
#
# # combine all pooled outputs
# total_filters = num_filter * len(filter_list)
# concat = mx.sym.Concat(*pooled_outputs, dim=1)
#
# # reshape for next layer
# h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))
#
# # dropout layer
# dropout=0.5
# print 'dropout probability', dropout
#
# if dropout > 0.0:
#     h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
# else:
#     h_drop = h_pool
#
# # fully connected layer
# num_label = len(labels_unique)
#
# cls_weight = mx.sym.Variable('cls_weight')
# cls_bias = mx.sym.Variable('cls_bias')
#
# fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)
#
# # softmax output
# sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')
#
# # set CNN pointer to the "back" of the network
# cnn = sm
#
#
# from collections import namedtuple
# import time
# import math
#
# # Define the structure of our CNN Model (as a named tuple)
# CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])
#
# # Define what device to train/test on
# ctx=mx.cpu(0)
# # If you have no GPU on your machine change this to
# # ctx=mx.cpu(0)
#
# arg_names = cnn.list_arguments()
#
# input_shapes = {}
# input_shapes['data'] = (batch_size, doc_size)
#
# arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
# arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
# args_grad = {}
# for shape, name in zip(arg_shape, arg_names):
#     if name in ['softmax_label', 'data']: # input, output
#         continue
#     args_grad[name] = mx.nd.zeros(shape, ctx)
#
# cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')
#
# param_blocks = []
# arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
# initializer=mx.initializer.Uniform(0.1)
# for i, name in enumerate(arg_names):
#     if name in ['softmax_label', 'data']: # input, output
#         continue
#     initializer(name, arg_dict[name])
#
#     param_blocks.append( (i, arg_dict[name], args_grad[name], name) )
#
# out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))
#
# data = cnn_exec.arg_dict['data']
# label = cnn_exec.arg_dict['softmax_label']
#
# cnn_model= CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)
#
#
# '''
# Train the cnn_model using back prop
# '''
#
# optimizer='rmsprop'
# max_grad_norm=5.0
# learning_rate=0.0005
# epoch=50
#
# print 'optimizer', optimizer
# print 'maximum gradient', max_grad_norm
# print 'learning rate (step size)', learning_rate
# print 'epochs to train for', epoch
#
# # create optimizer
# opt = mx.optimizer.create(optimizer)
# opt.lr = learning_rate
#
# updater = mx.optimizer.get_updater(opt)
#
# # create logging output
# logs = sys.stderr
#
# # For each training epoch
# for iteration in range(epoch):
#     tic = time.time()
#     num_correct = 0
#     num_total = 0
#
#     # Over each batch of training data
#     for begin in range(0, x_train.shape[0], batch_size):
#         batchX = x_train[begin:begin+batch_size]
#         batchY = y_train[begin:begin+batch_size]
#         print "batch Y: ", batchY
#         if batchX.shape[0] != batch_size:
#             continue
#         cnn_model.data[:] = batchX
#         cnn_model.label[:] = batchY
#
#         # forward
#         cnn_model.cnn_exec.forward(is_train=True)
#
#         # backward
#         cnn_model.cnn_exec.backward()
#
#         # eval on training data
#         num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
#         num_total += len(batchY)
#
#         # update weights
#         norm = 0
#         for idx, weight, grad, name in cnn_model.param_blocks:
#             grad /= batch_size
#             l2_norm = mx.nd.norm(grad).asscalar()
#             norm += l2_norm * l2_norm
#
#         norm = math.sqrt(norm)
#         for idx, weight, grad, name in cnn_model.param_blocks:
#             if norm > max_grad_norm:
#                 grad *= (max_grad_norm / norm)
#
#             updater(idx, grad, weight)
#
#             # reset gradient to zero
#             grad[:] = 0.0
#
#     # Decay learning rate for this epoch to ensure we are not "overshooting" optima
#     if iteration % 50 == 0 and iteration > 0:
#         opt.lr *= 0.5
#         print >> logs, 'reset learning rate to %g' % opt.lr
#
#     # End of training loop for this epoch
#     toc = time.time()
#     train_time = toc - tic
#     train_acc = num_correct * 100 / float(num_total)
#
#     # Saving checkpoint to disk
#     if (iteration + 1) % 10 == 0:
#         prefix = 'cnn'
#         cnn_model.symbol.save('./%s-symbol.json' % prefix)
#         save_dict = {('arg:%s' % k) :v  for k, v in cnn_model.cnn_exec.arg_dict.items()}
#         save_dict.update({('aux:%s' % k) : v for k, v in cnn_model.cnn_exec.aux_dict.items()})
#         param_name = './%s-%04d.params' % (prefix, iteration)
#         mx.nd.save(param_name, save_dict)
#         print >> logs, 'Saved checkpoint to %s' % param_name
#
#
#     # Evaluate model after this epoch on dev (test) set
#     num_correct = 0
#     num_total = 0
#
#     # For each test batch
#     for begin in range(0, x_test.shape[0], batch_size):
#         batchX = x_test[begin:begin+batch_size]
#         batchY = y_test[begin:begin+batch_size]
#
#         if batchX.shape[0] != batch_size:
#             continue
#
#         cnn_model.data[:] = batchX
#         cnn_model.cnn_exec.forward(is_train=False)
#
#         num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
#         num_total += len(batchY)
#
#     test_acc = num_correct * 100 / float(num_total)
#     print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
#             --- Test Accuracy thus far: %.3f' % (iteration, train_time, train_acc, test_acc)
