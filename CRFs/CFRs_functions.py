########### NER bio 2019/2020
# Mustapha BOUSSEBAINE
# Abderrahmane LARBI
# Bouchra ABIDAR
#########################
import nltk
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag, pos_tag_sents
import nltk.tag
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
import io
def corpus_preprocessing(path):
    pairwise = dict()
  
    paths = [path]
    sentences = []
    sentence = []
    p_token = ''

    all_length = []
    length = 0
    longer_entity = 0
    max_length = 0
    entity_count = 0
    labels_list = []
    tokens_list =[]
    tag_list =[]
    for path in paths:
        with open(path) as fin:
            reader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
            entity_length_turn_on = False
            for i, row in enumerate(reader):
                if i == 0:
                    pass
                if len(row) < 3:
                    sentences.append(sentence)
                    sentence = []
                else:
                    token = row[0]
                    label = row[1]
                    tag = row[2]

                    labels_list.append(label)
                    tag_list.append(tag)
                    tokens_list.append(token)
    return list(zip(tokens_list,tag_list, labels_list))

def Pos_tag_corpus(dataSet):
    training = pd.read_csv(dataSet,quoting=3, error_bad_lines=False,sep='\t')
    training.columns = ['token', 'tag']
    training = training.dropna()
    training = training.reset_index(drop=True)
    
    tokens = training['token'].tolist()
    tagged_texts = pos_tag_sents(map(word_tokenize, tokens))
    Pos =[]
    for i in range(len(tagged_texts)):
        Pos.append(tagged_texts[i][0][1])
    training['pos'] = pd.Series(Pos)
    return training

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def split_data_train_test(data_training, data_testing):
    ## training
    X_train = sent2features(data_training)
    X_train = [[i] for i in X_train]
    y_train = sent2labels(data_training)
    y_train = [[i] for i in y_train]
    ## testing
    X_test = sent2features(data_testing)
    X_test = [[i] for i in X_test]
    y_test = sent2labels(data_testing)
    y_test = [[i] for i in y_test]
    return X_train, y_train, X_test, y_test

def plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std):
    """ Plot a learning curve """

    # Plot training accuracy means for a given series of training sizes
    plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="Training accuracy")
    # Add a coloured fill showing the standard deviation of the training accuracy for a given series of training sizes
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
    # Plot test accuracy means for a given series of training sizes
    plt.plot(train_sizes, test_mean, color="green", linestyle="--", marker="s", markersize=5, label="Test accuracy")
    # Add a coloured fill showing the standard deviation of the test accuracy for a given series of training sizes
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
    # Add gridlines to the plot
    plt.grid()
    # Add captions to the X and Y axes of the plot
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    # Provide a location for the plot's legend/key
    plt.legend(loc="lower right")
    # Set upper and lower limits on the y axis
    plt.ylim([0.8, 1.0])
    # Show the plot
    plt.show()

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

def bio_classification_report(y_true, y_pred):
    
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )