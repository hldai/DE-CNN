import json
import numpy as np
import utils


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

        x = utils.label_sentence(sent_words, aspect_terms)
        labels_list.append(x)

    return labels_list, word_idxs_list


def __get_data(sents_file, words_list, word_idx_dict):

    with open(sents_file, encoding='utf-8') as f:
        sents = [json.loads(line) for line in f]

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
    return X, y


def __prep_dataset(train_sents_file, train_tok_text_file, test_sents_file, test_tok_text_file,
                   token_id_file, output_data_file, output_tokens_file):
    with open(token_id_file, encoding='utf-8') as f:
        word_idx_dict = json.loads(f.read())

    with open(train_tok_text_file, encoding='utf-8') as f:
        tok_texts = [line.strip() for line in f]
    words_list = [text.split(' ') for text in tok_texts]
    train_X, train_y = __get_data(train_sents_file, words_list, word_idx_dict)

    with open(test_tok_text_file, encoding='utf-8') as f:
        tok_texts = [line.strip() for line in f]
    words_list = [text.split(' ') for text in tok_texts]
    test_X, test_y = __get_data(test_sents_file, words_list, word_idx_dict)
    with open(output_tokens_file, 'w', encoding='utf-8') as fout:
        fout.write('{}\n'.format(json.dumps(words_list)))

    np.savez(output_data_file, train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)


def __gen_word_idx_file(tok_text_files, output_file):
    vocab = set()
    for filename in tok_text_files:
        with open(filename, encoding='utf-8') as f:
            for line in f:
                words = line.strip().split(' ')
                for w in words:
                    vocab.add(w)
    word_idx_dict = {w: i + 2 for i, w in enumerate(vocab)}
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write('{}\n'.format(json.dumps(word_idx_dict)))


token_id_file = 'data/prep_data/word_idx_dhl.json'

# __gen_word_idx_file(['d:/data/aspect/semeval14/laptops/laptops_train_texts_tokfc.txt',
#                      'd:/data/aspect/semeval14/laptops/laptops_test_texts_tokfc.txt',
#                      'd:/data/aspect/semeval14/restaurants/restaurants_train_texts_tokfc.txt',
#                      'd:/data/aspect/semeval14/restaurants/restaurants_test_texts_tokfc.txt',
#                      'd:/data/aspect/semeval15/restaurants/restaurants_train_texts_tokfc.txt',
#                      'd:/data/aspect/semeval15/restaurants/restaurants_test_texts_tokfc.txt'], token_id_file)

# train_sents_file = 'd:/data/aspect/semeval14/laptops/laptops_train_sents.json'
# train_tok_text_file = 'd:/data/aspect/semeval14/laptops/laptops_train_texts_tokfc.txt'
# test_sents_file = 'd:/data/aspect/semeval14/laptops/laptops_test_sents.json'
# test_tok_text_file = 'd:/data/aspect/semeval14/laptops/laptops_test_texts_tokfc.txt'
# output_file = 'data/prep_data/laptops14-dhl.npz'
# output_tokens_file = 'data/prep_data/laptops14-dhl-test-raw.json'
# __prep_dataset(train_sents_file, train_tok_text_file, test_sents_file, test_tok_text_file,
#                token_id_file, output_file, output_tokens_file)

train_sents_file = 'd:/data/aspect/semeval14/restaurants/restaurants_train_sents.json'
train_tok_text_file = 'd:/data/aspect/semeval14/restaurants/restaurants_train_texts_tokfc.txt'
test_sents_file = 'd:/data/aspect/semeval14/restaurants/restaurants_test_sents.json'
test_tok_text_file = 'd:/data/aspect/semeval14/restaurants/restaurants_test_texts_tokfc.txt'
output_file = 'data/prep_data/restaurants14-dhl.npz'
output_tokens_file = 'data/prep_data/restaurants14-dhl-test-raw.json'
__prep_dataset(train_sents_file, train_tok_text_file, test_sents_file, test_tok_text_file,
               token_id_file, output_file, output_tokens_file)

train_sents_file = 'd:/data/aspect/semeval15/restaurants/restaurants_train_sents.json'
train_tok_text_file = 'd:/data/aspect/semeval15/restaurants/restaurants_train_texts_tokfc.txt'
test_sents_file = 'd:/data/aspect/semeval15/restaurants/restaurants_test_sents.json'
test_tok_text_file = 'd:/data/aspect/semeval15/restaurants/restaurants_test_texts_tokfc.txt'
output_file = 'data/prep_data/restaurants15-dhl.npz'
output_tokens_file = 'data/prep_data/restaurants15-dhl-test-raw.json'
__prep_dataset(train_sents_file, train_tok_text_file, test_sents_file, test_tok_text_file,
               token_id_file, output_file, output_tokens_file)

# data = np.load(output_file)
# print(data['test_X'])
