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


def get_machine_name():
    import socket
    hostname = socket.gethostname()
    dot_pos = hostname.find('.')
    return hostname[:dot_pos] if dot_pos > -1 else hostname[:]
