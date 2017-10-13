#coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import sys
import random
import os
import zipfile

import numpy as np

import h5py

import tensorflow as tf


#def generate_sample(index_words, context_window_size):
#    """ Form training pairs according to the skip-gram model. """
#    for index in range(len(index_words)):
#        center  = index_words[index]
#        ctxt1 = random.randint(1, context_window_size)
#        # get a random target before the center word
#        for target in index_words[max(0, index - ctxt1): index]:
#            yield center, target
#        ctxt2 = random.randint(1, context_window_size)
#        # get a random target after the center wrod
#        for target in index_words[index + 1: index + ctxt2 + 1]:
#            yield center, target

            
def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for _ in range(len(index_words)):
        index   = np.random.randint(len(index_words))
        center  = index_words[index]
        ctxt1   = np.random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - ctxt1): index]:
            yield center, target
        ctxt2   = np.random.randint(1, context_window_size)
        # get a random target after the center wrod
        for target in index_words[index + 1: min(len(index_words),index + ctxt2 + 1)]:
            yield center, target

            
def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch


def process_data(npy_filename,vocab_size, batch_size, skip_window, max_npy_words=14000000):
    index_words = np.memmap(npy_filename,dtype='int64',mode='r+')
    # assert np.all(index_words < 50001), np.where(index_words >= 50001)
    
    index_words = index_words[:max_npy_words]
    print('loaded index_words: ',index_words.shape,file=sys.stderr)

    single_gen = generate_sample(index_words, skip_window)
    result = get_batch(single_gen, batch_size)
    return result

