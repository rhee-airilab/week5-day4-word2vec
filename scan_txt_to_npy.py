#!/usr/bin/env python
from __future__ import print_function, division
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import sys
import os
import time

import sqlite3
import numpy as np
from tqdm import tqdm

from remove_non_words import remove_non_words

#from konlpy.tag import Hannanum
#han = Hannanum()

from konlpy.tag import Komoran
han = Komoran()


# namuwiki_170327
# 14074786 input.txt

#$ wc text17.txt
#  14074786  188333610 1175894979 text17.txt

MAX_LINES = 14074786
MAX_WORDS = 188333610


def read_words_db(db_fn,max_words):
    """
    def read_words_db(db_fn,max_words):
    """
    with sqlite3.connect(db_fn) as db:
        c = db.cursor()
        c.execute('select * from words order by cnt desc limit {:d}'.format(max_words))
        words = c.fetchall()
        c.close()
        
        # make words index hash
        words_index = {}
        for i, pair in enumerate(words):
            w, c = pair
            w    = w.encode('utf-8')
            words_index[w] = i

        return words, words_index


def dump_words(words,output_tsv):
    """
    def dump_words(words,output_tsv):
    """
    with open(output_tsv,'w') as f:
        f.write('UNK|-1\n')
        for w,c in words:
            w = w.encode('utf-8')
            f.write('{:s}|{:d}\n'.format(w,c))


# XXX TBD how to read very very long one-line text file 
def scan_txt_to_npy(input_fn, words_index, output_npy):
    """
    def scan_txt_to_py(input_fn, words_index, output_npy):
    """
    pos = 0
    #out = np.lib.format.open_memmap(output_npy, 'write', dtype=np.int64, shape=[MAX_WORDS])
    out = np.memmap(output_npy, dtype=np.int64, mode='w+', shape=(MAX_WORDS,))

    with open(input_fn) as f:
        for _ in tqdm(range(MAX_LINES)):
            line = f.readline()
            if not line: break

            line = line.strip()
            if not line: continue

            for w in line.split(' '):
                w = w.decode('utf-8')
                if not w in words_index:
                    #logger.warning('scan_txt_to_py: unexpected word: {:s}'.format(w.encode('utf-8')))
                    pass
                else:
                    x = words_index[w]
                    out[pos] = x
                    pos += 1
                    if pos % 1000000 == 0:
                        tqdm.write('processed {:d} words'.format(pos))

    out.flush()
    assert pos < MAX_WORDS, ('pos < MAX',pos,MAX_WORDS)
    return pos


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_txt',nargs='+')
    parser.add_argument('--db',default='wordscount.db')
    parser.add_argument('--output_tsv')
    parser.add_argument('--output_npy')
    parser.add_argument('--max_words',type=int,default=50000)
    args = parser.parse_args()

    # read (word,cnt) tuples from db, descending order by count
    words, words_index = read_words_db(args.db,args.max_words)
    dump_words(words,args.output_tsv)

    # scan convert input txt into npy
    num_out_words = 0
    for fn in args.input_txt:
        w = scan_txt_to_npy(fn, words_index, args.output_npy)
        num_out_words += w


    print('############################################')
    print('# total words converted: {:d}'.format(num_out_words))
    print('############################################')
    

    # done
