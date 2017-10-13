# coding: utf-8
""" word2vec with NCE loss 
and code to visualize the embeddings on TensorBoard
"""

# # Tensorflow word2vec example
# 
# - based on CS 20SI: TensorFlow for Deep Learning Research. http://cs20si.stanford.edu
#   - https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/04_word2vec_visualize.py

from __future__ import print_function, division, absolute_import
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-'

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf


# === VOCAB_SIZE === 50001 (include UNK) # 100001 ( UNK 포함 )
VOCAB_SIZE = 50001
BATCH_SIZE = 128
EMBED_SIZE = 500  # 128  # dimension of the word embedding vectors
SKIP_WINDOW = 5  # the context window
NUM_SAMPLED = 10    # Number of negative examples to sample.
LEARNING_RATE = 0.1

NPY_FILENAME  = 'text17.npy'
MAX_NPY_WORDS = 188333610 - 100 # max words in textNN.npy

VOCAB_FILENAME = 'vocab_50k.tsv'
VOCAB_SEP      = '|'

NUM_TRAIN_STEPS = MAX_NPY_WORDS * 5
SKIP_STEP = 10 # how many steps to skip before reporting the loss


from process_data import process_data
make_batch_gen = lambda: process_data(NPY_FILENAME, VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, MAX_NPY_WORDS)


class SkipGramModel:
    """ Build the graph for word2vec model """
    def __init__(self,
                 vocab_size,
                 embed_size,
                 batch_size,
                 num_sampled,
                 learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name='global_step')

        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(
                tf.int32,
                shape=[self.batch_size],
                name='center_words')
            self.target_words = tf.placeholder(
                tf.int32,
                shape=[self.batch_size, 1],
                name='target_words')

        """ Step 2: define weights. In word2vec,
        it's actually the weights that we care about """
        with tf.name_scope("embed"):
            with tf.device('/cpu:0'):
                init_range = 0.5 / self.embed_size
                self.embed_matrix = tf.Variable(
                    tf.random_uniform(
                        [self.vocab_size, 
                         self.embed_size],
                        -init_range,
                        init_range), 
                    name='embed_matrix')
                
                # Step 3: define the inference
                embed = tf.nn.embedding_lookup(
                    self.embed_matrix,
                    self.center_words,
                    name='embed')

                # Step 4: define loss function
                # construct variables for NCE loss
                nce_weight = tf.Variable(
                    tf.truncated_normal(
                        [self.vocab_size, self.embed_size],
                        stddev=1.0 / (self.embed_size ** 0.5)), 
                    name='nce_weight')
                nce_bias = tf.Variable(
                    tf.zeros([VOCAB_SIZE]),
                    name='nce_bias')

        """ Step 3 + 4: define the model + the loss function """
        # define loss function to be NCE loss function
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weight, 
                biases=nce_bias, 
                labels=self.target_words, 
                inputs=embed, 
                num_sampled=self.num_sampled, 
                num_classes=self.vocab_size),
            name='loss')

        """ Step 5: define optimizer """
        #self.optimizer = \
        #     tf.train.GradientDescentOptimizer(self.lr). \
        #     minimize(self.loss,global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer(self.lr). \
           minimize(self.loss,global_step=self.global_step)

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("hist_center", self.center_words)
            tf.summary.histogram("hist_target", self.target_words)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def add_eval_graph(self):
        """Build the eval graph."""
        # Eval graph
        
        model = self

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        o_emb = model.embed_matrix  # without l2_normalize

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(o_emb, analogy_a)  # a's embs
        b_emb = tf.gather(o_emb, analogy_b)  # b's embs
        c_emb = tf.gather(o_emb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, o_emb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)
        
        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        model._analogy_a = analogy_a
        model._analogy_b = analogy_b
        model._analogy_c = analogy_c
        model._analogy_pred_idx = pred_idx


        
        # Nodes for computing neighbors for a given word according to
        # their cosine distance.

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        n_emb = tf.nn.l2_normalize(model.embed_matrix, 1)

        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(n_emb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, n_emb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(
            nearby_dist,
            min(1000, model.vocab_size))
        

        
        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        model._nearby_word = nearby_word
        model._nearby_val = nearby_val
        model._nearby_idx = nearby_idx


    def train_model(self, make_batch_gen, num_train_steps):
        model = self

        # defaults to saving all variables -
        #   in this case embed_matrix, nce_weight, nce_bias
        saver = tf.train.Saver()

        initial_step = 0
        
        config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options={'allow_growth': True,
                         'per_process_gpu_memory_fraction': 0.0})
        
        with tf.Session(config=config) as sess:
            cp = tf.train.latest_checkpoint('checkpoints')
            # if that checkpoint exists, restore from checkpoint
            if cp:
                saver.restore(sess, cp)
            else:
                tf.global_variables_initializer().run()

            # we use this to calculate late average loss
            #  in the last SKIP_STEP steps
            total_loss = 0.0
            
            writer = tf.summary.FileWriter(
                'improved_graph/lr' + str(LEARNING_RATE),
                sess.graph)

            initial_step = model.global_step.eval()

            batch_gen = make_batch_gen()

            for index in xrange(initial_step,
                                initial_step + num_train_steps):
                try:
                    centers, targets = batch_gen.next()
                except StopIteration:
                    logger.warn('batch_gen: end of batch. rewind')
                    batch_gen = make_batch_gen()
                    centers, targets = batch_gen.next()

                assert np.all(centers < VOCAB_SIZE), np.where(centers>=VOCAB_SIZE)
                assert np.all(targets < VOCAB_SIZE), np.where(targets>=VOCAB_SIZE)

                #logger.debug('step')

                feed_dict={
                    model.center_words: centers,
                    model.target_words: targets}

                loss_batch, _, summary = \
                    sess.run(
                        [model.loss,
                         model.optimizer,
                         model.summary_op], 
                        feed_dict=feed_dict)

                writer.add_summary(summary, global_step=index)

                total_loss += loss_batch

                if (index + 1) % SKIP_STEP == 0:
                    logger.debug('Average loss at step {}: {:5.1f}'. \
                          format(index+1,
                                 total_loss / SKIP_STEP))
                    total_loss = 0.0

                    saver.save(sess,
                               'checkpoints/skip-gram',
                               index+1)

            saver.save(sess,
                       'checkpoints/skip-gram',
                       index+1)



    def save_projection(self):

        model = self

        # ## 텐서보드 Projection

        ####################
        # code to visualize the embeddings. 
        final_embed_matrix = sess.run(model.embed_matrix)

        # it has to variable. constants don't work
        # here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(
            final_embed_matrix[:1000],
            name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # link this tensor to its metadata file, in 
        # this case the first 500 words of vocab
        embedding.metadata_path = VOCAB_FILENAME

        # saves a configuration file that TensorBoard 
        # will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model3.ckpt', 1)


class SkipGramQuery():
    def __init__(self,model):
        """
        """

        # create vocab index

        # vocab_dict, vocab_list = get_index_vocab(VOCAB_SIZE)

        with open(VOCAB_FILENAME) as f:
            vocab_list = [line.strip().split(VOCAB_SEP)[0]
                          for line
                          in f.xreadlines()]
        vocab_dict = {vocab_list[i]: i
                      for i
                      in range(len(vocab_list))}

        self.model      = model
        self.vocab_list = vocab_list
        self.vocab_dict = vocab_dict


    def get_word_index(self,word):
        return (self.vocab_dict[word]
                if word in self.vocab_dict
                else 0) # 0==UNK

    def print_words(self,ids,name='words'):
        print(name + ':')
        for x in ids:
            print(self.vocab_list[x], x)
        print()

    def query_abc(self,a_,b_,c_):
        """
        # solve c + (b - a)
        """
        print('=== query_abc ===')

        a, b, c = \
                [self.get_word_index(a_)], \
                [self.get_word_index(b_)], \
                [self.get_word_index(c_)]

        self.print_words(a,'a')
        self.print_words(b,'b')
        self.print_words(c,'c')

        sess = tf.get_default_session()

        preds = sess.run(
            self.model._analogy_pred_idx,
            {self.model._analogy_a:a,
             self.model._analogy_b:b,
             self.model._analogy_c:c})
        for i,pr_ in enumerate(preds):
            self.print_words(pr_,'c+b-a__'+str(i))

        return preds


    def query_likes(self,word_string):
        print('=== query_likes ===')

        a = [self.get_word_index(word_string)]

        self.print_words(a,'a')

        sess = tf.get_default_session()

        val, idx = sess.run(
            [self.model._nearby_val,
             self.model._nearby_idx],
            {self.model._nearby_word:a})

        for v,i in zip(val[0],idx[0])[:5]:
            self.print_words([i],'nearby_'+str(i)+', val='+str(v))

        return val, idx


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', nargs='?', const=True, default=False)
    args = parser.parse_args()

    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)

    model.add_eval_graph()

    config = tf.ConfigProto(gpu_options={'allow_growth':True})
    sess   = tf.InteractiveSession(config=config)

    if args.eval:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        saver  = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)

        model.save_projection()

        q = SkipGramQuery(model)

        # solve c + (b - a)
        q.query_abc('한국','서울','일본')
        # find nearest words
        q.query_likes('일본')

        # solve c + (b - a)
        q.query_abc('남자','여자','왕자')
        # find nearest words
        q.query_likes('제목')
    else:
        sess.run(tf.global_variables_initializer())
        model.train_model(make_batch_gen, NUM_TRAIN_STEPS)
        model.save_projection()

