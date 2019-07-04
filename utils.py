import numpy as np


def __find_sub_words_seq(words, sub_words):
    i, li, lj = 0, len(words), len(sub_words)
    while i + lj <= li:
        j = 0
        while j < lj:
            if words[i + j] != sub_words[j]:
                break
            j += 1
        if j == lj:
            return i
        i += 1
    return -1


def __label_words_with_terms(words, terms, label_val_beg, label_val_in, x):
    for term in terms:
        # term_words = term.lower().split(' ')
        term_words = term.split(' ')
        pbeg = __find_sub_words_seq(words, term_words)
        if pbeg == -1:
            print(words)
            print(terms)
            print()
            continue
        x[pbeg] = label_val_beg
        for p in range(pbeg + 1, pbeg + len(term_words)):
            x[p] = label_val_in


def label_sentence(words, aspect_terms):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    __label_words_with_terms(words, aspect_terms, label_val_beg, label_val_in, x)
    return x


def calc_f1(true_file, pred_file):
    sents_true = __get_sent_objs_se14_xml(true_file)
    sents_pred = __get_sent_objs_se14_xml(pred_file)

    def sents_to_dict(sents):
        sents_dict = dict()
        for sent in sents:
            sents_dict[sent['id']] = sent
        return sents_dict

    sents_dict_true = sents_to_dict(sents_true)
    sents_dict_pred = sents_to_dict(sents_pred)
    true_cnt, sys_cnt, hit_cnt = 0, 0, 0
    for sent_id, sent_true in sents_dict_true.items():
        sent_pred = sents_dict_pred[sent_id]
        # terms_true = [t['term'] for t in sent_true.get('terms', list())]
        terms_true = __get_terms_true(sent_true.get('terms', list()))
        terms_pred = [t['term'] for t in sent_pred.get('terms', list())]
        true_cnt += len(terms_true)
        sys_cnt += len(terms_pred)
        hit_cnt += __count_hit(terms_true, terms_pred)
    p, r, f1 = prf1(true_cnt, sys_cnt, hit_cnt)
    print(p, r, f1)
    return f1
