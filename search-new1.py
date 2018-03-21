import os
import math
import codecs
import jieba
import pickle
import nltk
import random
import math
from nltk.corpus import stopwords

# read articles
print("reading the articles")
articles = []
data_paths = ["raw_data/2013", "raw_data/2014"]
for dp in data_paths:
    for filename in os.listdir(dp):
        file_path = os.path.join(dp, filename)
        with codecs.open(file_path, "rb", "ISO-8859-1") as f:
            text = f.read()
        articles.append(text)

# generate the id for each article：id
article_map = dict(zip(range(len(articles)), articles))


# add the entities for splitting the text

with open("raw_data/labels/companies.csv", "rb") as f:
    companies = set([label.strip().decode().replace(",", " ").strip() for label in f.read().split(b"\r")])
    for l in companies:
        jieba.add_word(l)

with open("results/companies.txt", "rb") as f:
    for line in f:
        companies.add(line.strip().decode())
        jieba.add_word(line.strip().decode())

with open("raw_data/labels/ceo.csv", "rb") as f:
    ceos = set([label.strip().decode().replace(",", " ").strip() for label in f.read().split(b"\r")])
    for c in ceos:
        jieba.add_word(c)
    ceos.remove('Federal Reserve')

with open("results/ceo.txt", "rb") as f:
    for line in f:
        ceos.add(line.strip().decode())
        jieba.add_word(line.strip().decode())

jieba.add_word("GDP")


def cal_tfs(words):
    """
    calculate the word freq in a doc
    :param word:
    :param doc:
    :return:
    """
    t_freq = dict()
    for term in words:
        t_freq[term] = t_freq.get(term, 0) + 1
    for term in words:
        t_freq[term] = t_freq[term] / len(words)
    return t_freq


# build the inverse index
def create_index():
    #  the format of inverse index
    # {'word': [(1, 0.1), (2,0.01), (3,0.2)], 'the': [(7,0.03), (9,0.2)], 'active': [(1,0.7)]}
    print("begin creating index...")
    print("getting all words...")
    # find all words in all docs
    all_words = set()
    doc_words = dict()

    # calculate the doc freq for each word
    print("calculate the doc freq for each word...")
    word_dfs = {}
    for d_id, text in article_map.items():
        print(d_id)
        terms = [w.lower() for w in jieba.cut(text) if w not in ["", ' ', ',', '.', ':', ';', '?'] + stopwords.words('english')]
        doc_words[d_id] = terms
        all_words = all_words.union(set(terms))
        for term in set(terms):
            word_dfs[term] = word_dfs.get(term, 0) + 1

    print("building the index matrix...")
    index = {}
    for id, words in doc_words.items():
        word_tfs = cal_tfs(words)
        for word, tf in word_tfs.items():
            idf = math.log((len(doc_words) + 1) / (word_dfs[word] + 1))
            tf_ids = tf * idf
            if word in index:
                index[word].append((id, tf_ids))
            else:
                index[word] = []
    print("finish!")
    pickle.dump(index, open("model/index1.pkl", "wb"))

    return index


def search_index(keywords, has_index=True):
    keywords = [k.lower() for k in keywords]
    # load the inverse index
    if not has_index:
        index = create_index()
    else:
        index = pickle.load(open("model/index1.pkl", "rb"))

    if keywords:
        doc_scores = dict()
        for i, kw in enumerate(keywords):
            doc_ids = index.get(kw, [])
            for id, tf_idf in doc_ids:
                doc_scores[id] = doc_scores.get(id, [0, 0.0])
                doc_scores[id][0] += 1
                doc_scores[id][1] += tf_idf

        doc_scores = [(d, tup[1]) for d, tup in doc_scores.items() if tup[0] == len(keywords)]
        sorted_ids = sorted(doc_scores, key=lambda item: item[1], reverse=True)

        return [article_map[id] for id, score in sorted_ids]
    return []


def get_question_type(question):
    if "bankrupt" in question:
        return 1
    elif "CEO" in question:
        return 4
    elif "GDP" in question:
        return 2
    elif "percentage" in question:
        return 3
    else:
        return 0


def answer_question_one(question):
    month_dict = {
        1: ["January", "Jan."],
        2: ["February", "Feb."],
        3: ['March', 'Mar.'],
        4: ['April', 'Apr.'],
        5: ['May', 'May.'],
        6: ['June', 'Jun.'],
        7: ['July', 'Jul.'],
        8: ['August', 'Aug.'],
        9: ['September', 'Sep.'],
        10: ['October', 'Oct.'],
        11: ['November', 'Nov.'],
        12: ['December', 'Dec.']
    }
    year = question.split()[-1][:-1]
    month = question.split()[-4]
    int_m = -1
    try:
        int_m = int(month)
    except:
        pass
    answer = ""
    if int_m != -1:
        month = month_dict[int_m][0]
        key_words = ['bankrupt', year, month]
        docs = search_index(key_words, True)
        if len(docs) == 0:
            key_words[-1] = month_dict[int_m][-1]
            docs = search_index(key_words, True)

        answer = find_bankrupt_company(docs)

    return answer


def answer_question_four(question):
    """

    :param question
    :return:
    """
    pos = question.find("company")
    com = question[pos + len("company") + 1:-1].strip()
    key_words = ["CEO", com]
    docs = search_index(key_words)
    if len(docs) == 0:
        return "No Answer Found"
    return find_company_ceo(docs)


def answer_question_two(question):
    """

    :param question:
    :return:
    """
    key_words = ["GDP", "affect"]
    docs = search_index(key_words)

    answer = find_GDP_proprerty(docs)

    return answer

def answer_question_three(question):
    """

    :param question:
    :param proprety:
    :return:
    """

    key_words = ["GDP", "affect"]
    docs = search_index(key_words)
    pre = find_GDP_proprerty(docs)
    answer = find_GDP_percentage(docs, pre)
    return answer


def find_bankrupt_company(docs):
    """
    find the most occurs company in bankrupt article as the answer
    :param docs:
    :return:
    """
    com_counts = {c: 0 for c in companies}
    for doc in docs:
        for c in companies:
            com_counts[c] += doc.count(c)

    sorted_coms = sorted(com_counts.items(), key=lambda item: item[1], reverse=True)

    coms = [c for c, v in sorted_coms if v > 0]
    if len(coms) > 0:
        return random.choice(coms)

    return "No Answer Found"


def find_GDP_proprerty(docs):
    """
    find the proprety affect the GDP
    :param article:
    :return:
    """
    p_dict = dict()
    for doc in docs:
        sents = nltk.sent_tokenize(doc)
        # fine the sent with GDP
        sents = [sent for sent in sents
                 if 'GDP' in sent and ('%' in sent or 'percent' in sent or 'percentage' in sent)]
        for sent in sents:
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            for w, tag in tagged:
                if tag in ['NN', 'NNS', 'NNP', 'NNPS'] and len(w) > 5 and w not in ['GDP', 'percent', 'percentage']:
                    p_dict[w] = p_dict.get(w, 0) + 1

    sorted_p = sorted(p_dict.items(), key=lambda item: item[1], reverse=True)
    pros = [p for p, v in sorted_p if v > 10][:30]

    return random.choice(pros)


def find_GDP_percentage(docs, proprety):
    """

    :param docs:
    :return:
    """

    for doc in docs:
        sents = nltk.sent_tokenize(doc)
        # fine the sent with GDP
        sents = [sent for sent in sents
                 if 'GDP' in sent and ('%' in sent or 'percent' in sent or 'percentage' in sent)]
        for sent in sents:
            if proprety in sent:
                pos1 = sent.find("%")
                if pos1 != -1:
                    return sent[:pos1 + 1].split()[-1]

    return "No Answer Found!"

def find_company_ceo(docs):
    """
    find the name of ceo which is most near 'CEO' as the ceo of company
    :param docs:
    :param company:
    :return:
    """
    ceo_distances = {c: 0 for c in ceos}
    for doc in docs:
        l = 0
        r = len(doc)
        pos = doc.find('CEO', l, r)
        if pos != -1:
            for c in ceos:
                # before the pos
                c_pos1 = doc.find(c, l, pos)
                c_pos2 = doc[pos:].find(c)
                if c_pos1 != -1 and c_pos2 != -1:
                    ceo_distances[c] += min(pos - c_pos1, c_pos2 - pos)
                elif c_pos1 != -1 and c_pos2 == -1:
                    ceo_distances[c] += pos - c_pos1
                elif c_pos1 == -1 and c_pos2 != -1:
                    ceo_distances[c] += c_pos2 - pos
                else:
                    ceo_distances[c] += len(doc)

    sorted_ceos = sorted(ceo_distances.items(), key=lambda item: item[1])
    f_ceos = [c for c, v in sorted_ceos if len(c.split()) > 1]
    return f_ceos[0]


def main():
    while True:
        ques = input("please input the question, Q/q quit\n")
        q_type = get_question_type(ques)
        if q_type == 0:
            continue
        elif q_type == 1:
            answer = answer_question_one(ques)
            print(answer)
        elif q_type == 4:
            answer = answer_question_four(ques)
            print(answer)
        elif q_type == 2:
            answer = answer_question_two(ques)
            print(answer)
        elif q_type == 3:
            answer = answer_question_three(ques)
            print(answer)
        else:
            break


if __name__ == "__main__":
    main()
    # create_index()
