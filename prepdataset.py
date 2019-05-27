import json
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


def __get_word_idx_sequence(words_list, word_idx_dict, unk_id=1):
    seq_list = list()
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, unk_id) for w in words])
    return seq_list


def data_from_sents_file(sents, words_list, word_idx_dict):
    word_idxs_list = __get_word_idx_sequence(words_list, word_idx_dict)
    len_max = max([len(words) for words in words_list])
    print('max sentence len:', len_max)

    labels_list = list()
    for sent_idx, (sent, sent_words) in enumerate(zip(sents, words_list)):
        aspect_objs = sent.get('terms', list())
        aspect_terms = [t['term'] for t in aspect_objs]

        x = label_sentence(sent_words, aspect_terms)
        labels_list.append(x)

    return labels_list, word_idxs_list


def __prep_dataset(sents_file, tok_text_file, token_id_file, output_data_file, output_tokens_file):
    with open(token_id_file, encoding='utf-8') as f:
        word_idx_dict = json.loads(f.read())

    with open(tok_text_file, encoding='utf-8') as f:
        tok_texts = [line.strip() for line in f]

    with open(sents_file, encoding='utf-8') as f:
        sents = [json.loads(line) for line in f]

    words_list = [text.split(' ') for text in tok_texts]
    with open(output_tokens_file, 'w', encoding='utf-8') as fout:
        fout.write('{}\n'.format(json.dumps(words_list)))

    labels_list, word_idxs_list = data_from_sents_file(sents, words_list, word_idx_dict)
    max_sent_len = 83
    X = np.zeros((len(word_idxs_list), max_sent_len), np.int32)
    y = np.zeros((len(labels_list), max_sent_len), np.int32)

    for i, word_idxs in enumerate(word_idxs_list):
        for j, widx in enumerate(word_idxs):
            X[i][j] = widx
    for i, labels in enumerate(labels_list):
        for j, l in enumerate(labels):
            y[i][j] = l
    # data = dict()
    # data['test_X'] = X
    # data['test_y'] = y
    np.savez(output_file, test_X=X, test_y=y)


sents_file = 'd:/data/aspect/semeval14/laptops/laptops_test_sents.json'
tok_text_file = 'd:/data/aspect/semeval14/laptops/laptops_test_texts_tokfc.txt'
token_id_file = 'data/prep_data/word_idx.json'
output_file = 'data/prep_data/laptops14-dhl-test.npz'
output_tokens_file = 'data/prep_data/laptops14-dhl-test-raw.json'
__prep_dataset(sents_file, tok_text_file, token_id_file, output_file, output_tokens_file)
# data = np.load(output_file)
# print(data['test_X'])
