#!/usr/bin/env python
from __future__ import print_function, division

import sys
import os
import logging
import time

from tqdm import tqdm

#from konlpy.tag import Hannanum
#han = Hannanum()

from konlpy.tag import Komoran
han = Komoran()

import sqlite3

from remove_non_words import remove_non_words

# namuwiki_170327
# 14074786 input.txt

MAX_LINES = 14074786


def parse_words(input_fn,output_file,db):

    c = db.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS words
                 (word text primary key, cnt integer)''')

    commit_interval = 100000
    t_start = time.time()
    words_passed = words_passed_start = 0

    with open(input_fn) as f:
        # while True:
        max_lines = MAX_LINES
        for _ in tqdm(range(max_lines)):
            line = f.readline()
            if not line: break
            line = remove_non_words(line).decode('utf-8',errors='ignore')
            elems = han.pos(line, flatten=True)

            for e in elems:

                print(e[0].encode('utf-8'),end=' ',file=output_file)

                w = e[0] + u'/' + e[1]

                c.execute('INSERT OR IGNORE INTO words VALUES (?, 0)',(w,))
                c.execute('UPDATE words SET cnt = cnt + 1 WHERE word = ?',(w,))

                words_passed += 1

                if (words_passed) % commit_interval == 0:
                    t_now = time.time()
                    mean_passed = 1.0 * (words_passed - words_passed_start) / (t_now - t_start + 1e-5)
                    tqdm.write('words_passwd: {}, mean_passed {:.2f}'.format(words_passed,mean_passed))
                    t_start = t_now
                    words_passed_start = words_passed

            #print()

    print('close',file=sys.stderr)
    c.execute('select * from words order by cnt desc limit 100')
    ans = c.fetchall()
    for a in ans:
        print(a[0].encode('utf-8'),a[1],file=sys.stderr)
    c.close()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files',nargs='+')
    parser.add_argument('-o','--output_file',default='text17.txt')
    parser.add_argument('-w','--word_db',default='wordscount.db')
    args = parser.parse_args()

    with sqlite3.connect(args.word_db) as db:
        for fn in args.input_files:
            with open(args.output_file,'w') as output_file:
                parse_words(fn,output_file,db)
