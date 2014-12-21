import json
from nltk.stem.snowball import RussianStemmer
from string import punctuation
import re
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize


class Solution:
    def __init__(self):
        self.vocabulary = {}
        self.characteristics = {}
        self.num_of_classes = 39
        self.classifiers = []
        for i in range(self.num_of_classes):
            self.classifiers.append(OneVsRestClassifier(MultinomialNB(alpha=1.0), n_jobs=-1))
            #self.classifiers.append(OneVsRestClassifier(MultinomialNB(alpha=1.0)))

    def text_preprocess(self, text, train=True):
        text.lower()
        for punct in punctuation:
            text = text.replace(punct, ' ')
        text = re.sub('\s+', ' ', text) # replace multiple whitespaces with one ' '
        tokens = text.split(' ')
        tokens = filter(lambda x: len(x) > 3, tokens) # remove stop-words
        stemmer = RussianStemmer()
        tokens = map(stemmer.stem, tokens)
        if train:
            for token in tokens:
                if token not in self.vocabulary.keys():
                    self.vocabulary[token] = len(self.vocabulary)
        return tokens

    def get_features(self, tokens):
        features = np.array([0] * len(self.vocabulary))
        for token in tokens:
            token_index = self.vocabulary.get(token, -1)
            if token_index != -1:
                features[token_index] += 1
        return features

    def train(self, training_corpus):
        texts = training_corpus[0]
        opinions = training_corpus[1]

        #prepare texts
        all_tokens = []
        for text in texts:
            all_tokens.append(self.text_preprocess(text))

        # getting features from texts
        features = []
        for one_text_tokens in all_tokens:
            features.append(self.get_features(one_text_tokens))

        # getting targets from opinions
        targets = []
        for opinion in opinions:
            for characteristic in opinion:
                if characteristic not in self.characteristics.keys():
                    self.characteristics[characteristic] = len(self.characteristics)

            targets.append((map(lambda x: self.characteristics[x], opinion)))

        targets = label_binarize(targets, multilabel=True, classes=range(len(self.characteristics)))

        # training
        for i in range(self.num_of_classes):
            self.classifiers[i].fit(features, targets)

    def getClasses(self, texts):
        classes = []

        for text in texts:
            tokens = self.text_preprocess(text, train=False)
            features = self.get_features(tokens)

            one_text_prediction = []
            for i in range(self.num_of_classes):
                one_text_prediction.append(self.classifiers[i].predict(features)[0])
            prediction = np.array(one_text_prediction).mean(0)
            answer = []
            for i in range(len(prediction)):
                if prediction[i] >= 0.5:
                    answer.append(1)
                else:
                    answer.append(0)
            answ_characteristics = []
            for (characteristic, index) in self.characteristics.items():
                if answer[index] == 1:
                    answ_characteristics.append(characteristic)

            classes.append(answ_characteristics)

        return classes


def get_train_data(filename):
    datafile = open(filename, 'r')
    data = json.load(datafile)
    datafile.close()

    texts = []
    opinions = []

    for response in data:
        texts.append(response['text'])

        opinion = []
        for response_values in response['answers']:
            for characteristic in response_values.keys():
                if characteristic != 'text':
                    opinion.append((response_values['text'], characteristic))

        opinions.append(opinion)

    return (texts, opinions)

if __name__ == '__main__':
    training_corpus = get_train_data('reviews.json')
    solution = Solution()
    solution.train((training_corpus[0][:-10], training_corpus[1][:-10]))
    texts = training_corpus[0][-10:]
    classes = solution.getClasses(texts)
    i = 1
    for one_class in classes:
        print "response " + str(i)
        i += 1
        for characteristic in one_class:
            print characteristic[0], characteristic[1]
