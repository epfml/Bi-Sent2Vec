/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_ARGS_H
#define FASTTEXT_ARGS_H

#include <istream>
#include <ostream>
#include <string>

namespace fasttext {

enum class model_name : int {cbow=1, transgram, supi, bisent2vec};
enum class loss_name : int {hs=1, ns, softmax};

class Args {
  public:
    Args();
    std::string input;
    std::string test;
    std::string output;
    std::string dict;
    double lr;
    double boostNgrams;
    int lrUpdateRate;
    int dim;
    int ws;
    int dropoutK;
    int epoch;
    int maxVocabSize;
    int minCount;
    int minCountLabel;
    int neg;
    int wordNgrams;
    int numCheckPoints;
    loss_name loss;
    model_name model;
    int bucket;
    int bucketChar;
    int minn;
    int maxn;
    int thread;
    double t;
    std::string label;
    int verbose;
    std::string pretrainedVectors;
    int saveOutput;

    bool qout;
    bool retrain;
    bool qnorm;
    size_t cutoff;
    size_t dsub;

    void parseArgs(int, char**);
    void printHelp();
    void save(std::ostream&);
    void load(std::istream&);
};

}

#endif
