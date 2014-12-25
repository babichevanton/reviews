import json
from nltk.stem.snowball import RussianStemmer
from string import punctuation
import re
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import metrics


class Solution:
    def __init__(self):
        self.vocabulary = {}
        self.characteristics = {}
        self.num_of_classes = 39
        self.classifiers = []
        for i in range(self.num_of_classes):
            self.classifiers.append(MultinomialNB(alpha=1.0))

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

    def filter_train(self, training_corpus):
        responses = {}
        texts = training_corpus[0]
        opinions = training_corpus[1]

        # get statistics of poll
        for ind in range(len(texts)):
            if not (texts[ind] in responses):
                responses[texts[ind]] = {}
            for opinion_ind in range(len(opinions[ind])):
                if opinions[ind][opinion_ind] in responses[texts[ind]]:
                    responses[texts[ind]][opinions[ind][opinion_ind]] += 1
                else:
                    responses[texts[ind]][opinions[ind][opinion_ind]] = 1

        # get max persons voted
        max_voted_list = []
        for text in responses.keys():
            max_voted = 0
            for opinion in responses[text].keys():
                if responses[text][opinion] > max_voted:
                    max_voted = responses[text][opinion]
            max_voted_list.append(max_voted)

        # filter training corpus
        filtered_texts = responses.keys()
        opinions = [(max_voted, opinion) for (max_voted, (text, opinion)) in zip(max_voted_list, responses.items())]
        filtered_opinions = []
        for ind in range(len(filtered_texts)):
            max_voted = opinions[ind][0]
            # choose all characteristics that were estimated in the same way by all persons
            filtered_opinion = [opinion for (opinion, voted) in opinions[ind][1].items() if (voted == max_voted)]
            filtered_opinions.append(filtered_opinion)

        #return training_corpus
        return (filtered_texts, filtered_opinions)

    def get_features(self, tokens):
        features = np.array([0] * len(self.vocabulary))
        for token in tokens:
            token_index = self.vocabulary.get(token, -1)
            if token_index != -1:
                features[token_index] += 1
        return features

    def get_targets(self, opinions, class_index):
        # getting global targets from opinions
        targets_global = []
        for opinion in opinions:
            for characteristic in opinion:
                if characteristic not in self.characteristics.keys():
                    self.characteristics[characteristic] = len(self.characteristics)

            targets_global.append((map(lambda x: self.characteristics[x], opinion)))

        targets = []
        cur_target = 0
        for text_targets in targets_global:
            for target in text_targets:
                if target == class_index:
                    cur_target = 1
                    break
            targets.append(cur_target)

        return targets

    def train(self, json_data):
        #training_corpus = self.filter_train(get_train_data(json_data))
        training_corpus = json_data

        texts = training_corpus[0][:]
        opinions = training_corpus[1][:]

        #prepare texts
        all_tokens = []
        for text in texts:
            all_tokens.append(self.text_preprocess(text))

        # getting features from texts
        features = []
        for one_text_tokens in all_tokens:
            features.append(self.get_features(one_text_tokens))

        # training
        # "i" classifier determines "i" class
        for i in range(self.num_of_classes):
            targets = self.get_targets(opinions, i)
            self.classifiers[i].fit(features, targets)

    def getClasses(self, texts):
        # construct feature vectors
        features = []
        for text in texts:
            tokens = self.text_preprocess(text, train=False)
            text_features = self.get_features(tokens)
            features.append(text_features)

        # prediction
        predictions = []
        for i in range(self.num_of_classes):
            #predictions.append(self.classifiers[i].predict(features))
            one_prediction_proba = self.classifiers[i].predict_proba(features)
            predictions.append(one_prediction_proba)

        # get list of characteristics
        characteristic_list = []
        for i in range(len(self.characteristics)):
            for item in self.characteristics.items():
                if i == item[1]:
                    characteristic_list.append(item[0])

        # get answer from predictions
        classes = [ [] for i in range(len(texts))]
        #classes_proba = [ [] for i in range(len(texts))]
        for characteristic in range(len(predictions)):
            for text_ind in range(len(predictions[characteristic])):
                #classes_proba[text_ind].append(predictions[characteristic][text_ind])
                if (characteristic < len(characteristic_list)) and (predictions[characteristic][text_ind][0] >= 0.5):
                    classes[text_ind].append(characteristic_list[characteristic])

        #return classes_proba
        return classes

    def score(self, texts, opinions):
        features = []
        for text in texts:
            tokens = self.text_preprocess(text, train=False)
            text_features = self.get_features(tokens)
            features.append(text_features)

        for i in range(self.num_of_classes):
            targets = self.get_targets(opinions, i)
            one_prediction_score = self.classifiers[i].score(features, targets)
            print one_prediction_score

    def compute_ft(self, training_corpus):

        features = []
        for text in texts:
            tokens = self.text_preprocess(text, train=False)
            text_features = self.get_features(tokens)
            features.append(text_features)

        for i in range(self.num_of_classes):
            targets = self.get_targets(opinions, i)
            one_prediction_score = self.classifiers[i].score(features, targets)
            print one_prediction_score


def get_train_data(json_data):
    texts = []
    opinions = []

    for response in json_data:
        texts.append(response['text'])

        opinion = []
        for response_values in response['answers']:
            for characteristic in response_values.keys():
                if characteristic != 'text':
                    opinion.append((response_values['text'], characteristic))

        opinions.append(opinion)

    return (texts, opinions)


if __name__ == '__main__':
    trainfile = open('reviews.json', 'r')
    json_data = json.load(trainfile)
    trainfile.close()
    solution = Solution()
    training_corpus = get_train_data(json_data[:])
    texts_train, texts_test, opinions_train, opinions_test = cross_validation.train_test_split(training_corpus[0],
                                                                                               training_corpus[1],
                                                                                               test_size=0.4,
                                                                                               random_state=0)
    solution.train((texts_train, opinions_train))
    classes = solution.score(texts_test, opinions_test)
    solution.compute_ft(training_corpus)
