from platform import platform
import datetime
import logging
import os
import torch
import numpy as np
import random
import json
import utils
import datautils
from loggingutils import init_logging
from decnn import batch_generator, Model

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


def __get_numpy_data(word_idx_seqs, label_seqs, max_sent_len):
    X = np.zeros((len(word_idx_seqs), max_sent_len), np.int32)
    y = np.zeros((len(label_seqs), max_sent_len), np.int32)

    for i, word_idxs in enumerate(word_idx_seqs):
        for j, widx in enumerate(word_idxs):
            X[i][j] = widx
    for i, labels in enumerate(label_seqs):
        for j, l in enumerate(labels):
            y[i][j] = l
    return X, y


def get_data(tok_texts_file, tokfc_texts_file, terms_file, token_id_file):
    unk_id = 1
    max_sent_len = 83
    terms_true_list = datautils.load_json_objs(terms_file)

    with open(token_id_file, encoding='utf-8') as f:
        word_idx_dict = json.loads(f.read())

    word_idx_seqs, label_seqs = list(), list()
    f_tok = open(tok_texts_file, encoding='utf-8')
    f_tokfc = open(tokfc_texts_file, encoding='utf-8')
    for i, (line_tok, line_tokfc) in enumerate(zip(f_tok, f_tokfc)):
        words_tok = line_tok.strip().split(' ')
        if len(words_tok) > max_sent_len or len(words_tok) == 0:
            continue

        words_tokfc = line_tokfc.strip().split(' ')
        if len(words_tok) != len(words_tokfc):
            continue

        word_idx_seq = [word_idx_dict.get(w, unk_id) for w in words_tokfc]
        aspect_terms = terms_true_list[i]

        label_seq = utils.label_sentence(words_tok, aspect_terms)

        word_idx_seqs.append(word_idx_seq)
        label_seqs.append(label_seq)
        # if i > 10:
        #     break

    perm = np.random.permutation(len(word_idx_seqs))
    n_train = len(word_idx_seqs) - 2000
    idxs_valid = set(perm[n_train:])

    train_word_idx_seqs, dev_word_idx_seqs = list(), list()
    train_label_seqs, dev_label_seqs = list(), list()

    for i in range(len(word_idx_seqs)):
        if i in idxs_valid:
            dev_word_idx_seqs.append(word_idx_seqs[i])
            dev_label_seqs.append(label_seqs[i])
        else:
            train_word_idx_seqs.append(word_idx_seqs[i])
            train_label_seqs.append(label_seqs[i])

    X_train, y_train = __get_numpy_data(train_word_idx_seqs, train_label_seqs, max_sent_len)
    X_dev, y_dev = __get_numpy_data(dev_word_idx_seqs, dev_label_seqs, max_sent_len)
    return X_train, y_train, X_dev, y_dev


def valid_loss(model, valid_X, valid_y, crf=False):
    model.eval()
    losses = []
    for batch in batch_generator(valid_X, valid_y, crf=crf):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask = batch
        loss = model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_y)
        # print(loss.data.cpu().numpy())
        # losses.append(loss.data[0])
        losses.append(loss.data.cpu().numpy())
    model.train()
    return sum(losses) / len(losses)


def __pretrain(gen_emb_file, domain_emb_file, tok_texts_file, tokfc_texts_file, terms_file, token_id_file,
               n_epochs, batch_size, lr=0.0001, dropout=0.5, use_crf=False, output_file=None):
    X_train, y_train, X_dev, y_dev = get_data(tok_texts_file, tokfc_texts_file, terms_file, token_id_file)
    logging.info('{} samples, {} batches'.format(X_train.shape[0], X_train.shape[0] // batch_size))
    # exit()
    gen_emb = np.load(gen_emb_file)
    domain_emb = np.load(domain_emb_file)

    model = Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
    model.cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)

    best_loss = float("inf")
    i_batch = 0
    losses_train = list()
    for epoch in range(n_epochs):
        for batch in batch_generator(X_train, y_train, batch_size, crf=use_crf):
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask = batch
            loss = model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y)
            losses_train.append(loss.data.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
            # loss = valid_loss(model, train_X, train_y, crf=use_crf)
            i_batch += 1
            if i_batch % 1000 == 0:
                loss = valid_loss(model, X_dev, y_dev, crf=use_crf)
                # print(losses_train)
                logging.info('{} {} l_t={:.4f} l_v={:.4f}'.format(epoch, i_batch, sum(losses_train), loss))
                losses_train = list()
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model, output_file)
                    logging.info('model saved to {}'.format(output_file))
        # shuffle_idx = np.random.permutation(len(X_train))
        # X_train = X_train[shuffle_idx]
        # y_train = y_train[shuffle_idx]


if __name__ == "__main__":
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    if platform().startswith('Windows'):
        model_dir = 'd:/data/aspect/decnndata/'
        data_dir = 'd:/data/aspect'
        amazon_dir = 'd:/data/res/amazon/'
    else:
        model_dir = '/home/hldai/data/aspect/decnndata/'
        data_dir = '/home/hldai/data/aspect'
        amazon_dir = '/home/hldai/data/res/amazon/'

    gen_emb_file = 'data/prep_data/glove.840B.300d.txt.npy'
    domain_emb_file = 'data/prep_data/laptop_emb.vec.npy'

    tok_texts_file = os.path.join(amazon_dir, 'laptops-reivews-sent-tok-text.txt')
    tokfc_texts_file = os.path.join(amazon_dir, 'laptops-reivews-sent-text-tokfc.txt')
    terms_file = os.path.join(data_dir, 'semeval14/laptops/amazon-laptops-aspect-rm-rule-result.txt')
    token_id_file = 'data/prep_data/word_idx.json'
    output_model_file = os.path.join(data_dir, 'decnndata/laptops-decnn.pth')
    n_epochs = 35
    batch_size = 128
    __pretrain(gen_emb_file, domain_emb_file, tok_texts_file, tokfc_texts_file, terms_file, token_id_file,
               n_epochs, batch_size, output_file=output_model_file)
