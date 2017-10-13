# coding: utf-8
from __future__ import print_function, division

import sys
import time
import re

import numpy as np

from tqdm import tqdm

from konlpy.tag import Komoran
han = Komoran()

def read_vocab(dict_file='data/vocab_100k.txt'):
    t_start = time.time()
    dictionary = dict()
    index = 0
    with open(dict_file) as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip()
            pos,cnt = line.split('|')
            cnt = int(cnt)
            dictionary[pos] = index
            # word,tag = pos.split('/')
            index += 1
    t_elapsed = time.time() - t_start
    print('read_vocab(): elapsed {:.1f}'.format(t_elapsed),file=sys.stderr)
    return dictionary

# CHUNK_SIZE=5000
MAX_WORDS=94900000

#   1806747  45674879 490322255 data/text11.txt
def words_to_index(input_file='data/text11.txt',output_file='data/text11.npy',dictionary=None):
    if not dictionary:
        dictionary = read_vocab()
    #out = h5py.File(output_file)
    #out.create_dataset('words',(CHUNK_SIZE,1),chunks=(CHUNK_SIZE,1),maxshape=(None,1),dtype='int64')
    words = np.memmap(output_file,dtype='int64',mode='w+',shape=(MAX_WORDS,))
    words_count = 0
    t_start = time.time()
    with open(input_file) as f:
        # while True:
        for line_index in tqdm(range(1806747)):
            line = f.readline()
            if not line: break
            line = line.strip().decode('utf-8',errors='ignore')
            elems = han.pos(line, flatten=True)
            # discard rest lines if max_words limit would be hit
            if words_count + len(elems) >= MAX_WORDS:
                words.flush()
                break
            for e in elems:
                w = ( e[0] + u'/' + e[1] ).encode('utf-8')
                idx = dictionary[w] if w in dictionary else 0
                if idx > 0:
                    words[words_count] = idx
                    words_count += 1
                    if words_count % 10000 == 0:
                        words.flush()
                        t_elapsed = time.time() - t_start
                        tqdm.write('line: {}, words: {}, elapsed: {:.1f}'.format(line_index+1,words_count,t_elapsed))
                        t_start = time.time()
    words.flush()
    print('=== TOTAL WORDS ===',words_count,file=sys.stderr)
    with open('total_words.csv','w') as f: f.write(str(words_count)+'\n')

    """
    got: === TOTAL WORDS === 94899892
    """

if __name__ == '__main__':
    words_to_index()


