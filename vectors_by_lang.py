import numpy as np
import codecs
import sys, getopt
import csv
import os
import gzip, pickle
import sys
from scipy import stats
import argparse

def divide_vectors(vec_file):
    file_contents=open(vec_file,'r',encoding='utf-8-sig')
    vectors = list()
    id2word = dict()
    num_vec = dict()
    word_count = 0
    for i,line in enumerate(file_contents):
        entries = line.split(" ",1)
        if i==0:
            num_words = int(entries[0])
            dim = int(entries[1])
        else:
            if "_" not in entries[0]:
                continue
            word = entries[0][:-3]
            lang = entries[0][-2:]
            id2word[len(vectors)] = (lang,word)
            if lang not in num_vec:
                num_vec[lang]=0
            num_vec[lang]+=1
            vectors.append(entries[1])
        if i%100000==0:
            print(str(i) + " words loaded")
    file_contents.close()

    file_contents_output = dict()
    print("Writing vectors")

    for lang in num_vec:
        file_contents_output[lang] = open(vec_file[:-4] + "_" + lang + ".vec",'w',encoding='utf-8-sig')
        file_contents_output[lang].write(str(num_vec[lang]) + " " + str(dim) + "\n")

    for i,vector in enumerate(vectors):
        file_contents_output[id2word[i][0]].write(id2word[i][1] + " " + vector)

    for lang in num_vec:
        file_contents_output[lang].close()

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for vector separation by language')

    parser.add_argument('--vector_file', action='store', type=str,
                        help='vector file location')

    args = parser.parse_args()

    divide_vectors(args.vector_file)
