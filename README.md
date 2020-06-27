# Bi-Sent2Vec

TLDR: This library provides cross-lingual numerical representations (features) for words, short texts, or sentences, which can be used as input to any machine learning task with applications geared towards cross-lingual word translation, cross-lingual sentence retrieval as well as cross-lingual downstream NLP tasks. The library is a cross-lingual extension of [Sent2Vec](https://github.com/epfml/sent2vec).

Bi-Sent2Vec vectors are also well suited to monolingual tasks as indicated by a marked improvement in the monolingual quality of the word embeddings. (For more details, see [paper](https://arxiv.org/abs/1912.12481))

### Table of Contents

* [Setup and Requirements](#setup-and-requirements)
* [Using the model](#using-the-model)
    - [Downloading Bi-Sent2Vec pre-trained vectors](#downloading-bi-sent2vec-pre-trained-vectors)
    - [Train a New Bi-Sent2Vec Model](#train-a-new-bi-sent2vec-model)
* [Evaluation](#evaluation)
* [References](#references)

# Setup and Requirements

Our code builds upon [Facebook's FastText library](https://github.com/facebookresearch/fastText).

To compile the library, simply run the `make` command.

# Using the model

For the purpose of generating cross-lingual word and sentence representations, we introduce our Bi-Sent2vec method and provide code and models.

The method uses a simple but efficient  objective to train distributed representations of sentences. The algorithm outperforms the current state-of-the-art bag-of-words based models on most of the benchmark tasks, and is also competitive with deep models on some of the tasks, highlighting the robustness of the produced word and  sentence embeddings, see [*the paper*](https://arxiv.org/abs/1912.12481) for more details.

## Downloading Bi-Sent2Vec pre-trained vectors

Models trained and tested in the Bi-Sent2Vec paper can be downloaded from the following links. Users are encouraged to add more bi-lingual models to the list provided they have been benchmarked properly.

### Unigram

[EN](https://drive.google.com/file/d/1schNkg0OLTrTqA_VSCcpJnczaZbEfUiW/view?usp=sharing)-[DE](https://drive.google.com/file/d/1S76Pf_UByF9vHfGHx3EAP5bB3Vvyi8_l/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1b_q6WCXdQEKz0Grx21mzxVaGqBV7Y5WY/view?usp=sharing)-[ES](https://drive.google.com/file/d/1pEusR2238oJwLmRzC0j7pduaKW6FLzOv/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1Omac6Cbkb7cmyOeTZpyacGOKGHy9ixo8/view?usp=sharing)-[FI](https://drive.google.com/file/d/1rr_ZhDPjp901vGKUuK4gjXEM9aDBDOOD/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1Ny7TDW_1jRZTH327OhrGbSpIPgsSr3LJ/view?usp=sharing)-[FR](https://drive.google.com/file/d/1WTsLmVcjG_M8vwgUvfvM_A1386q7Nv0H/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1dPmM270pUTW2ETl14SfcFFI0hscEeQXO/view?usp=sharing)-[HU](https://drive.google.com/file/d/1aLe8CsB2o0fjmMTonYozcj0V69pPY59_/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1C6e-6qkhsoYjWlJ0OcjWeStOSUvQUOuQ/view?usp=sharing)-[IT](https://drive.google.com/file/d/1_rO75UgpZpug7kzjtho-jkigwl_igFqm/view?usp=sharing)

### Bigram

[EN](https://drive.google.com/file/d/1CI4sFR0Y6v6zHzdaFN17vn9Wo_m1ywno/view?usp=sharing)-[DE](https://drive.google.com/file/d/1HyKS0QpBHd_2pLp_0JxGFDAMASydR5Xe/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1XKk4Vw4ATMcYAhmDl_nUx1HnpIcfuEwX/view?usp=sharing)-[ES](https://drive.google.com/file/d/1oJ2LXUk0CZzwj02sWIVZtsXH6psICOHl/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1q9dn76Sau3ArOEJ-J2mYghAnnoPjb9us/view?usp=sharing)-[FI](https://drive.google.com/file/d/1cqen99e_BNZp13wWGBpJf7x5b-PmHEKl/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1ztCsll3YUUBVMDHTZBPDmNMcZPkQbEu-/view?usp=sharing)-[FR](https://drive.google.com/file/d/1KWuuFpNDOmEXoLvwWU2OastlK5MyIke6/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/15sMJQNm3s6uWh80y2SkxY6hCWnMx5pmx/view?usp=sharing)-[HU](https://drive.google.com/file/d/1K88rEsVM7mrcHZlqwHPvtjWy5JPkIvJ6/view?usp=sharing)
•
[EN](https://drive.google.com/file/d/1Iv-vuPWw40mvkbzfRJ7c0EIvvO_VRjT6/view?usp=sharing)-[IT](https://drive.google.com/file/d/1XXDGKQFscr_snJFzGc-aafUVRy_q6r0g/view?usp=sharing)


## Train a New Bi-Sent2Vec Model
### Tokenizing and data format
Bi-Sent2Vec requires parallel sentences (sentences which are translations of each other) for training.
We use [spacy](https://spacy.io/) tokenizer to tokenize the text.

The required data format is one sentence pair per line. The two parallel sentences are separated by a \<\<split\>\> token and each word has its language code attached to it as a prefix. For example, here is an example of a snapshot of a valid English-French dataset -
```
the_en train_en is_en arriving_en ._en <<split>> le_fr train_fr arrive_fr ._fr
france_en won_en the_en world_en cup_en ._en <<split>> la_fr france a_fr gagné_fr la_fr coupe_fr du_fr monde_fr ._fr
```

## Training

Assuming en-fr_sentences.txt is the pre-processed training corpus, here is an example of a command to train a Bi-Sent2Vec model:

    ./fasttext bisent2vec -input en-fr_sentences.txt -output model-en-fr -dim 300 -lr 0.2 -neg 10 -bucket 2000000 -maxVocabSize 750000 -thread 30 -t 0.000005 -epoch 5 -minCount 8 -dropoutK 4 -loss ns -wordNgrams 2 -numCheckPoints 5

Here is a description of all available arguments:

```
The following arguments are mandatory:
  -input              training file path
  -output             output file path (model is stored in the .bin file and the vectors in .vec file)

The following arguments are optional:
  -lr                 learning rate [0.2]
  -lrUpdateRate       change the rate of updates for the learning rate [100]
  -dim                dimension of word and sentence vectors [100]
  -epoch              number of epochs [5]
  -minCount           minimal number of word occurences [5]
  -minCountLabel      minimal number of label occurences [0]
  -neg                number of negatives sampled [10]
  -wordNgrams         max length of word ngram [2]
  -loss               loss function {ns, hs, softmax} [ns]
  -bucket             number of hash buckets for vocabulary [2000000]
  -thread             number of threads [2]
  -t                  sampling threshold [0.0001]
  -dropoutK           number of ngrams dropped when training a Bi-Sent2Vec model [2]
  -verbose            verbosity level [2]
  -maxVocabSize       vocabulary exceeding this size will be truncated [None]
  -numCheckPoints     number of intermediary checkpoints to save when training [1]
```
### Post Processing
Use vectors_by_lang.py to separate the vectors for the two different languages.
Example -
```
python vectors_by_lang.py model-en-fr.vec en fr
```
This code will create two files model-en-fr_en.vec and model-en-fr_fr.vec in word2vec format containing vectors for English and French respectively.

# Evaluation
Our models are evaluated using the standard evaluation tool in the [MUSE](https://github.com/facebookresearch/MUSE) repository by Facebook AI Research.

# References
When using this code or some of our pretrained vectors for your application, please cite the following paper:

  Ali Sabet, Prakhar Gupta, Jean-Baptiste Cordonnier, Robert West, Martin Jaggi [*Robust Cross-lingual Embeddings from Parallel Sentences*](https://arxiv.org/abs/1912.12481)

```
@article{Sabet2019RobustCE,
  title={Robust Cross-lingual Embeddings from Parallel Sentences},
  author={Ali Sabet and Prakhar Gupta and Jean-Baptiste Cordonnier and Robert West and Martin Jaggi},
  journal={ArXiv 1912.12481},
  year={2020},
}
```
