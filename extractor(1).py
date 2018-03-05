import os
import codecs
import pickle
import datetime
import nltk
from nltk.corpus import stopwords
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import re

# define entity label
entity_label = {
    "ceo": "O",
    "companies": "C",
    "percentage": "P",
    "other": "I"
}

# 1. split the text into sentences


def sent_tokenize(text):
    """
    split the text into sentences
    :param text: the original text
    :return: sentences in list
    """
    sents = nltk.sent_tokenize(text)
    return sents


def get_sentences():
    """
    get sentences from data directory
    :return: all sentences in all text
    """
    data_path1 = "raw_data/2013"
    data_path2 = "raw_data/2014"
    sentences = []

    for filename in os.listdir(data_path1):
        filepath = os.path.join(data_path1, filename)
        with codecs.open(filepath, "rb", "ISO-8859-1") as f:
            text = f.read()
        sentences.extend(sent_tokenize(text))

    for filename in os.listdir(data_path2):
        filepath = os.path.join(data_path2, filename)
        with codecs.open(filepath, "rb", "ISO-8859-1") as f:
            text = f.read()
        sentences.extend(sent_tokenize(text))
    return sentences


def read_labels(label_name):
    """

    :param label_name:
    :return:
    """
    label_path = "raw_data/labels/" + label_name + ".csv"
    if label_name == "companies":
        with open(label_path, "rb") as f:
            labels = [label.strip().decode() for label in f.read().split(b"\r")]
        post_labels = []
        for lb in labels:
            if lb[-1] == '.':
                post_labels.append(lb[:-1])
            else:
                post_labels.append(lb)
        return set(post_labels)
    elif label_name == "ceo":
        with open(label_path, "rb") as f:
            labels = [label.strip().replace(b"\r", b" ").decode()
                      for label in f.read().split(b",") if len(label.strip()) > 0]
        return set(labels)

    elif label_name == "percentage":
        with open(label_path, "rb") as f:
            labels = set([label.strip().decode() for label in f.read().split(b"\r")
                          if (b'%' in label or b'percentage' in label or b'percent' in label)])
        return set(labels)


# 2. build the train and test set
def build_train_test_set(sentences, label_name):
    """
    build train and test set based on sentences
    :param sentences: sentences extracted from text
    :param label_name: the label name
    :return: train and test set
    """
    # 2.1 read the words with label
    labels = read_labels(label_name)

    # 2.2 find the sentences with label words
    train_sentences = []
    test_sentences = []

    # used to get the real words of entity
    original_sentences = []

    for k, sent in enumerate(sentences):
        has_label = False
        cur_label = ""
        for label in labels:
            if label in sent:
                cur_label = label
                has_label = True
                break

        # cut the sentences into words and delete the stop words
        # if label_name == "percentage":
        #     punctuation = '!,;:?"\''
        #     sent = re.sub(r'[{}]+'.format(punctuation), ' ', sent)
        #     tokens = [w for w in sent.split() if len(w) > 0]
        # else:
        tokens = [w for w in nltk.word_tokenize(sent) if w not in stopwords.words('english')]
        # add the pos tag of words
        tagged = nltk.pos_tag(tokens)
        label_words = cur_label.split()

        if has_label:
            doc = []
            for i, token in enumerate(tokens):
                mark = entity_label.get("other")
                if token in label_words:
                    mark = entity_label.get(label_name)
                doc.append((token, tagged[i][1], mark))
            train_sentences.append(doc)
        else:
            original_sentences.append(sent)
            doc = []
            for i, token in enumerate(tokens):
                doc.append((token, tagged[i][1]))
            test_sentences.append(doc)

    return train_sentences, test_sentences, original_sentences


# 3. extract features from word
def word2feature(sent, i):
    """
    define features in word
    :param word: a word in sentence
    :return: features
    """
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),  # the word itself
        "word[-3:]": word[-3:],  # the postfix of the word with length 3
        "word[-2:]": word[-2:],  # the postfix of the word with length 2
        "word.isupper": word.isupper(),  # the word is upper or not
        "word.istitle": word.istitle(),  # the first Character of word is upper or not
        "word.isdigit()": word.isdigit(),  # the word is digit or not
        "postag": postag,  # the part of speech of the word
        "postag[:2]": postag[:2] # the postfix of word's pos
    }

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
            "-1:postag": postag1,
            "-1:postag[:2]": postag1[:2]
        })
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word1.isupper(),
            "+1:postag": postag1,
            "+1:postag[:2]": postag1[:2]
        })
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    """
    convert the sentence into features
    :param sent: the sentence
    :return:
    """
    return [word2feature(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for w, postag, label in sent]


# 4. build model
def train(train_set, e_type):
    """
    training the data to build model
    :param train_set:
    :return:
    """
    X = [sent2features(s) for s in train_set]
    y = [sent2labels(s) for s in train_set]
    X_train, X_eva, y_train, y_eva = sklearn.model_selection.train_test_split(X, y)
    # crf model
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # evaluation
    labels = list(crf.classes_)
    y_pred = crf.predict(X_eva)
    print("The model's performance on evaluation set for " + e_type + " entity")
    print(metrics.flat_classification_report(y_eva, y_pred, labels=labels, digits=4))

    # store the model
    with open("model/" + e_type + "_crf.pkl", "wb") as f:
        pickle.dump(crf, f)


# 5. extract other entities
def extract_entity(pred, sent, ori_sent, e_tag):
    """
    extarct entity from a predicted sent
    :param pred: the predict result
    :param sent: the sentence
    :param ori_sent: the original sentence
    :return: str, the entity
    """
    c_pos = []
    for i, tag in enumerate(pred):
        if tag == e_tag:
            c_pos.append(i)

    if len(c_pos) == 1:
        return {sent[c_pos[0]][0]}

    tups = []
    m = 0
    n = 0
    k = 1
    while k < len(c_pos):
        while k < len(c_pos) and c_pos[k] - c_pos[n] < 2:
            n += 1
            k += 1
        tups.append((m, n))
        m = k
        n = m
        k = m + 1

    entities = set()
    f_index = 0
    for tup in tups:
        if tup[0] == tup[1]:
            entities.add(sent[c_pos[tup[0]]][0])
        else:
            first_word = sent[c_pos[tup[0]]][0]
            end_word = sent[c_pos[tup[1]]][0]

            f_index = ori_sent.find(first_word, f_index)
            e_index = ori_sent.find(end_word, f_index) + len(end_word)
            entities.add(ori_sent[f_index:e_index])
            f_index = e_index
    return entities


def predict(test_sents, original_sents, e_tag, l_name):
    """
    predict entities from test sentences
    :param test_sents:
    :param original_sents:
    :return:
    """
    X_test = [sent2features(s) for s in test_sents]
    # load the model
    crf = pickle.load(open("model/" + l_name + "_crf.pkl", "rb"))
    y_test = crf.predict(X_test)
    entities = set()
    for i, y in enumerate(y_test):
        if e_tag in y:
            entities = entities.union(extract_entity(y, test_sents[i], original_sents[i], e_tag))
    return entities


def search_percent(sent):
    """
    search the percentage by re
    :param sent:
    :return:
    """
    pattern = "%|percentage|percent"
    c_ptn = re.compile(pattern)
    entities = set()
    p_start = 0
    while True:
        pos = c_ptn.search(sent, p_start)

        if pos is not None:
            if pos.group() == "%":
                entities.add(sent[p_start:pos.end()].split(" ")[-1])
            else:
                tokens = sent[p_start:pos.end()].split(" ")
                if len(tokens) > 2:
                    entities.add(tokens[-2] + " " + tokens[-1])

            p_start = pos.end()
        else:
            break

    return entities



def run_company():
    start_time = datetime.datetime.now()

    # 1. build the train and test set for extracting the companies
    # print("start building the train and test set for companies entity.")
    # sentences = get_sentences()
    # train_data, test_data, original_data = build_train_test_set(sentences, 'companies')
    mid_time1 = datetime.datetime.now()
    # print("building finished")
    # print("it costs %d s to build the companies train and test data set" % (mid_time1 - start_time).seconds)

    # store the train and test set, so we can just load data set instead of building again when we use it 
    # pickle.dump(train_data, open("post_data/companies_train_set.pkl", "wb"))
    # pickle.dump(test_data, open("post_data/companies_test_set.pkl", "wb"))
    # pickle.dump(original_data, open("post_data/companies_original_set.pkl", "wb"))
#
    # 2. build the modle
    # load the train set
    print("start building the crf model")
    train_sents = pickle.load(open("post_data/companies_train_set.pkl", "rb"))
    train(train_sents, "companies")
    mid_time2 = datetime.datetime.now()
    print("training finished")
    print("it costs %d s to training the model" % (mid_time2 - mid_time1).seconds)

    # 3. extract the entities in test set
    # load the test set
    print("start predicting the unknown companies entities")
    test_sents = pickle.load(open("post_data/companies_test_set.pkl", "rb"))
    original_sents = pickle.load(open("post_data/companies_original_set.pkl", "rb"))
    entities = predict(test_sents, original_sents, 'C', 'companies')
    with open("results/companies.txt", "wt") as f:
        f.write("\n".join(entities))
    mid_time3 = datetime.datetime.now()
    print("predicting finished")
    print("it costs %d s to extract unknown companies entitie" % (mid_time3 - mid_time2).seconds)
    print("the results in 'results/companies.txt'")

def run_ceo():
    # start_time = datetime.datetime.now()

    # 1. build the train and test set for extracting the companies
    # print("start building the train and test set for ceo entity.")
    # sentences = get_sentences()
    # train_data, test_data, original_data = build_train_test_set(sentences, 'ceo')
    mid_time1 = datetime.datetime.now()
    # print("building finished")
    # print("it costs %d s to build the ceo train and test data set" % (mid_time1 - start_time).seconds)
    #
    # # store the train and test set, so we can just load data set instead of building again when we use it
    # pickle.dump(train_data, open("post_data/ceo_train_set.pkl", "wb"))
    # pickle.dump(test_data, open("post_data/ceo_test_set.pkl", "wb"))
    # pickle.dump(original_data, open("post_data/ceo_original_set.pkl", "wb"))

    # 2. build the modle
    # load the train set
    print("start building the crf model")
    train_sents = pickle.load(open("post_data/ceo_train_set.pkl", "rb"))
    train(train_sents, "ceo")
    mid_time2 = datetime.datetime.now()
    print("training finished")
    print("it costs %d s to training the model" % (mid_time2 - mid_time1).seconds)

    # 3. extract the entities in test set
    # load the test set
    print("start predicting the unknown ceo entities")
    test_sents = pickle.load(open("post_data/ceo_test_set.pkl", "rb"))
    original_sents = pickle.load(open("post_data/ceo_original_set.pkl", "rb"))
    entities = predict(test_sents, original_sents, 'O', "ceo")
    with open("results/ceo.txt", "wt") as f:
        f.write("\n".join(entities))
    mid_time3 = datetime.datetime.now()
    print("predicting finished")
    print("it costs %d s to extract unknown ceo entitie" % (mid_time3 - mid_time2).seconds)
    print("the results in 'results/ceo.txt'")


def run_percentage():
    start_time = datetime.datetime.now()

    # 1. build the train and test set for extracting the companies
    # print("start building the train and test set for percentage entity.")
    # sentences = get_sentences()
    # train_data, test_data, original_data = build_train_test_set(sentences, 'percentage')
    mid_time1 = datetime.datetime.now()
    # print("building finished")
    # print("it costs %d s to build the percentage train and test data set" % (mid_time1 - start_time).seconds)

    # store the train and test set, so we can just load data set instead of building again when we use it
    # pickle.dump(train_data, open("post_data/percentage_train_set.pkl", "wb"))
    # pickle.dump(test_data, open("post_data/percentage_test_set.pkl", "wb"))
    # pickle.dump(original_data, open("post_data/percentage_original_set.pkl", "wb"))

    # 2. build the modle
    # load the train set
    print("start building the crf model")
    train_sents = pickle.load(open("post_data/percentage_train_set.pkl", "rb"))
    train(train_sents, "percentage")
    mid_time2 = datetime.datetime.now()
    print("training finished")
    print("it costs %d s to training the model" % (mid_time2 - mid_time1).seconds)

    # 3. extract the entities in test set
    # load the test set
    print("start predicting the unknown percentage entities")
    test_sents = pickle.load(open("post_data/percentage_test_set.pkl", "rb"))
    original_sents = pickle.load(open("post_data/percentage_original_set.pkl", "rb"))
    entities = predict(test_sents, original_sents, 'P', "percentage")
    re_entities = set()
    for sent in original_sents:
        re_entities = re_entities.union(search_percent(sent))
    entities = entities.union(re_entities)
    with open("results/percentage.txt", "wt") as f:
        f.write("\n".join(entities))
    mid_time3 = datetime.datetime.now()
    print("predicting finished")
    print("it costs %d s to extract unknown percentage entitie" % (mid_time3 - mid_time2).seconds)
    print("the results in 'results/percentage.txt'")


def main():
    run_company()

    run_ceo()

    run_percentage()


if __name__ == "__main__":
    main()
