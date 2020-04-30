/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "args.h"

#include <stdlib.h>
#include <string.h>

#include <iostream>

namespace fasttext {

Args::Args() {
  lr = 0.05;
  boostNgrams = 1.0;
  dim = 100;
  ws = 5;
  dropoutK = 0;
  epoch = 5;
  minCount = 5;
  minCountLabel = 0;
  neg = 5;
  wordNgrams = 1;
  loss = loss_name::ns;
  model = model_name::transgram;
  bucket = 2000000;
  bucketChar = 1;
  minn = 0;
  maxn = 0;
  thread = 12;
  lrUpdateRate = 100;
  t = 1e-4;
  label = "__label__";
  verbose = 2;
  pretrainedVectors = "";
  saveOutput = 0;
  maxVocabSize = -1;
  numCheckPoints = 1;
  qout = false;
  retrain = false;
  qnorm = false;
  cutoff = 0;
  dsub = 2;
}

void Args::parseArgs(int argc, char** argv) {
  std::string command(argv[1]);
  if (command == "bisent2vec") {
    model = model_name::bisent2vec;
    loss = loss_name::ns;
    neg = 10;
    minCount = 5;
    minn = 3;
    maxn = 6;
    lr = 0.05;
    dropoutK = 2;
  }
  int ai = 2;
  while (ai < argc) {
    if (argv[ai][0] != '-') {
      std::cout << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    if (strcmp(argv[ai], "-h") == 0) {
      std::cout << "Here is the help! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    } else if (strcmp(argv[ai], "-input") == 0) {
      input = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-test") == 0) {
      test = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-output") == 0) {
      output = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dict") == 0) {
      dict = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lr") == 0) {
      lr = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-boostNgrams") == 0) {
      boostNgrams = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lrUpdateRate") == 0) {
      lrUpdateRate = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dim") == 0) {
      dim = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-ws") == 0) {
      ws = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-epoch") == 0) {
      epoch = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minCount") == 0) {
      minCount = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minCountLabel") == 0) {
      minCountLabel = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-neg") == 0) {
      neg = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-numCheckPoints") == 0) {
      numCheckPoints = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dropoutK") == 0) {
      dropoutK = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-wordNgrams") == 0) {
      wordNgrams = atoi(argv[ai + 1]);
      if (wordNgrams == 1) bucket = 1;
    } else if (strcmp(argv[ai], "-loss") == 0) {
      if (strcmp(argv[ai + 1], "hs") == 0) {
        loss = loss_name::hs;
      } else if (strcmp(argv[ai + 1], "ns") == 0) {
        loss = loss_name::ns;
      } else if (strcmp(argv[ai + 1], "softmax") == 0) {
        loss = loss_name::softmax;
      } else {
        std::cout << "Unknown loss: " << argv[ai + 1] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } else if (strcmp(argv[ai], "-bucket") == 0) {
      bucket = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-bucketChar") == 0) {
      bucketChar = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minn") == 0) {
      minn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-maxn") == 0) {
      maxn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-thread") == 0) {
      thread = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-t") == 0) {
      t = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-label") == 0) {
      label = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-verbose") == 0) {
      verbose = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-maxVocabSize") == 0) {
      maxVocabSize = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-pretrainedVectors") == 0) {
      pretrainedVectors = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-saveOutput") == 0) {
      saveOutput = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-qnorm") == 0) {
      qnorm = true; ai--;
    } else if (strcmp(argv[ai], "-retrain") == 0) {
      retrain = true; ai--;
    } else if (strcmp(argv[ai], "-qout") == 0) {
      qout = true; ai--;
    } else if (strcmp(argv[ai], "-cutoff") == 0) {
    cutoff = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dsub") == 0) {
      dsub = atoi(argv[ai + 1]);
    } else {
      std::cout << "Unknown argument: " << argv[ai] << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    ai += 2;
  }
  if (input.empty() || output.empty()) {
    std::cout << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  if (wordNgrams <= 1 && maxn == 0) {
    bucket = 0;
  }
}

void Args::printHelp() {
  std::string lname = "ns";
  if (loss == loss_name::hs) lname = "hs";
  if (loss == loss_name::softmax) lname = "softmax";
  std::cout
    << "\n"
    << "The following arguments are mandatory:\n"
    << "  -input              training file path\n"
    << "  -output             output file path\n\n"
    << "The following arguments are optional:\n"
    << "  -lr                 learning rate [" << lr << "]\n"
    << "  -lrUpdateRate       change the rate of updates for the learning rate [" << lrUpdateRate << "]\n"
    << "  -dim                size of word vectors [" << dim << "]\n"
    << "  -ws                 size of the context window [" << ws << "]\n"
    << "  -epoch              number of epochs [" << epoch << "]\n"
    << "  -minCount           minimal number of word occurences [" << minCount << "]\n"
    << "  -minCountLabel      minimal number of label occurences [" << minCountLabel << "]\n"
    << "  -neg                number of negatives sampled [" << neg << "]\n"
    << "  -wordNgrams         max length of word ngram [" << wordNgrams << "]\n"
    << "  -loss               loss function {ns, hs, softmax} [ns]\n"
    << "  -bucket             number of buckets [" << bucket << "]\n"
    << "  -maxVocabSize       vocabulary exceeding this size will be truncated [None]\n"
    << "  -numCheckPoints     number of intermediary checkpoints to save when training [" << numCheckPoints << "]\n"
    << "  -minn               min length of char ngram [" << minn << "]\n"
    << "  -maxn               max length of char ngram [" << maxn << "]\n"
    << "  -thread             number of threads [" << thread << "]\n"
    << "  -t                  sampling threshold [" << t << "]\n"
    << "  -label              labels prefix [" << label << "]\n"
    << "  -dropoutK           number of ngrams dropped when training a sent2vec model [" << dropoutK << "]\n"
    << "  -verbose            verbosity level [" << verbose << "]\n"
    << "  -pretrainedVectors  pretrained word vectors for supervised learning []\n"
    << "  -saveOutput         whether output params should be saved [" << saveOutput << "]\n"
    << "\nThe following arguments for quantization are optional:\n"
    << "  -cutoff             number of words and ngrams to retain [" << cutoff << "]\n"
    << "  -retrain            finetune embeddings if a cutoff is applied [" << retrain << "]\n"
    << "  -qnorm              quantizing the norm separately [" << qnorm << "]\n"
    << "  -qout               quantizing the classifier [" << qout << "]\n"
    << "  -dsub               size of each sub-vector [" << dsub << "]\n"
    << std::endl;
}

void Args::save(std::ostream& out) {
  out.write((char*) &(dim), sizeof(int));
  out.write((char*) &(ws), sizeof(int));
  out.write((char*) &(epoch), sizeof(int));
  out.write((char*) &(minCount), sizeof(int));
  out.write((char*) &(neg), sizeof(int));
  out.write((char*) &(wordNgrams), sizeof(int));
  out.write((char*) &(loss), sizeof(loss_name));
  out.write((char*) &(model), sizeof(model_name));
  out.write((char*) &(bucket), sizeof(int));
  out.write((char*) &(minn), sizeof(int));
  out.write((char*) &(maxn), sizeof(int));
  out.write((char*) &(lrUpdateRate), sizeof(int));
  out.write((char*) &(t), sizeof(double));
}

void Args::load(std::istream& in) {
  in.read((char*) &(dim), sizeof(int));
  in.read((char*) &(ws), sizeof(int));
  in.read((char*) &(epoch), sizeof(int));
  in.read((char*) &(minCount), sizeof(int));
  in.read((char*) &(neg), sizeof(int));
  in.read((char*) &(wordNgrams), sizeof(int));
  in.read((char*) &(loss), sizeof(loss_name));
  in.read((char*) &(model), sizeof(model_name));
  in.read((char*) &(bucket), sizeof(int));
  in.read((char*) &(minn), sizeof(int));
  in.read((char*) &(maxn), sizeof(int));
  in.read((char*) &(lrUpdateRate), sizeof(int));
  in.read((char*) &(t), sizeof(double));
}

}
