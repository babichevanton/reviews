import json
import numpy as np
from string import punctuation, digits

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn import cross_validation
from sklearn import metrics


class TargetExtractor:
    def __init__(self, num_of_classes):
        self.all_chars = {}
        self.all_inds = {}
        self.target_len = num_of_classes

    def fit(self, opinions):
        for opinion in opinions:
            for char in opinion:
                if char not in self.all_chars:
                    new_ind = len(self.all_chars)
                    self.all_chars[char] = new_ind
                    self.all_inds[new_ind] = char
        return self

    def transform(self, opinions):
        targets = []
        for opinion in opinions:
            one_target = np.array([0] * self.target_len)
            ind_found = 0
            ind_notfound = len(self.all_chars)
            for char in opinion:
                if char in self.all_chars:
                    one_target[ind_found] = self.all_chars[char]
                    ind_found += 1
                else:
                    one_target[ind_notfound] = ind_notfound
                    ind_notfound += 1
            targets.append(one_target)
        return np.array(targets)

    def inverse_transform(self, targets):
        opinions = []
        for one_target in targets:
            opinion = []
            for ind in one_target:
                if ind in self.all_inds:
                    opinion.append(self.all_inds[ind])
            opinions.append(opinion)
        return opinions

    def fit_transform(self, opinions):
        return self.fit(opinions).transform(opinions)


class Solution:
    def __init__(self):
        self.num_of_classes = 39
        self.v = CountVectorizer(ngram_range=(1,2))
        self.le = TargetExtractor(self.num_of_classes)
        self.classifier = OneVsRestClassifier(LinearSVC(tol=1e-5))

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

    def text_preprocess(self, text, train=True):
        for punct in punctuation:
            text = text.replace(punct, ' ')
        for digit in digits:
            text = text.replace(digit, '')
        return text

    def train(self, json_data):
        #training_corpus = self.filter_train(get_train_data(json_data))
        training_corpus = json_data

        texts = training_corpus[0]
        opinions = training_corpus[1]

        for text_ind in range(len(texts)):
            texts[text_ind] = self.text_preprocess(texts[text_ind])

        features = self.v.fit_transform(texts);
        targets = self.le.fit_transform(opinions)

        self.classifier.fit(features, targets)

    def getClasses(self, texts):
        features = self.v.transform(texts);
        predictions = self.classifier.predict(features)
        classes = self.le.inverse_transform(predictions)
        return classes


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
                                                                                               test_size=0.3,
                                                                                               random_state=1)
    
    solution.train((texts_train, opinions_train))
    res = solution.getClasses(texts_test)
    resv = solution.le.transform(res)
    opv = solution.le.transform(opinions_test)

    score = metrics.f1_score(resv, opv, average='micro')
    print score
