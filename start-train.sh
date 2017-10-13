#!/bin/sh

#jupyter nbconvert --to python nb-word2vec-visualize.ipynb

#rm    -fr processed improved_graph checkpoints
#mkdir -p  processed improved_graph checkpoints

CUDA_VISIBLE_DEVICES=2 python nb-word2vec-visualize.py  2>&1 | tee -a train.log 
