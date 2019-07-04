import argparse
import torch
import os
import xml.etree.ElementTree as ET
import time
import json
import numpy as np
import math
import random
from decnn import batch_generator, Model

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


def label_terms(sent_texts, token_seqs, label_seqs):
    terms_list = list()
    for i, sent_tokens in enumerate(token_seqs):
        lb = label_seqs[i]
        token_idx, pt, tag_on = 0, 0, False
        start, end = -1, -1
        sent_text = sent_texts[i]
        terms = list()
        for ix, c in enumerate(sent_text):
            if token_idx < len(sent_tokens) and pt >= len(sent_tokens[token_idx]):
                pt = 0
                token_idx += 1

            if token_idx < len(sent_tokens) and lb[token_idx] == 1 and pt == 0 and c != ' ':
                if tag_on:
                    end = ix
                    tag_on = False
                    terms.append(sent_text[start:end])
                start = ix
                tag_on = True
            elif token_idx < len(sent_tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on:
                start = ix
                tag_on = True
            elif token_idx < len(sent_tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0:
                end = ix
                tag_on = False
                term = sent_text[start:end]
                terms.append(sent_text[start:end])
                # if not term.endswith(sent_tokens[token_idx - 1]):
                #     print(term, sent_tokens[token_idx - 1])
            elif token_idx >= len(sent_tokens) and tag_on:
                end = ix
                tag_on = False
                terms.append(sent_text[start:end])
            if c == ' ' or ord(c) == 160:
                pass
            elif sent_tokens[token_idx][pt:pt+2] == '``' or sent_tokens[token_idx][pt:pt+2] == "''":
                pt += 2
            else:
                pt += 1
        if tag_on:
            tag_on = False
            end = len(sent_text)
            terms.append(sent_text[start:end])
        terms_list.append(terms)
        # print(sent_text, terms)
    return terms_list


def test_dhl(model, sent_texts, test_X, token_seqs, batch_size=128, crf=False):
    pred_y = np.zeros((test_X.shape[0], 83), np.int16)
    model.eval()
    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len = np.sum(test_X[offset:offset + batch_size] != 0, axis=1)
        batch_idx = batch_test_X_len.argsort()[::-1]
        batch_test_X_len = batch_test_X_len[batch_idx]
        batch_test_X_mask = (test_X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_test_X = test_X[offset:offset+batch_size][batch_idx]
        batch_test_X_mask = torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long().cuda())
        batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long().cuda())
        batch_pred_y = model(batch_test_X, batch_test_X_len, batch_test_X_mask, testing=True)
        r_idx = batch_idx.argsort()
        if crf:
            batch_pred_y = [batch_pred_y[idx] for idx in r_idx]
            for ix in range(len(batch_pred_y)):
                for jx in range(len(batch_pred_y[ix])):
                    pred_y[offset+ix, jx] = batch_pred_y[ix][jx]
        else:
            batch_pred_y = batch_pred_y.data.cpu().numpy().argmax(axis=2)[r_idx]
            pred_y[offset:offset + batch_size, :batch_pred_y.shape[1]] = batch_pred_y
    model.train()
    assert len(pred_y) == len(test_X)
    terms_list = label_terms(sent_texts, token_seqs, pred_y)
    return terms_list


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


def __count_hit(terms_true, terms_pred):
    terms_true, terms_pred = terms_true.copy(), terms_pred.copy()
    terms_true.sort()
    terms_pred.sort()
    idx_pred = 0
    cnt_hit = 0
    for t in terms_true:
        while idx_pred < len(terms_pred) and terms_pred[idx_pred] < t:
            idx_pred += 1
        if idx_pred == len(terms_pred):
            continue
        if terms_pred[idx_pred] == t:
            cnt_hit += 1
            idx_pred += 1
    return cnt_hit


def prf1(n_true, n_sys, n_hit):
    p = n_hit / (n_sys + 1e-6)
    r = n_hit / (n_true + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    return p, r, f1


def __get_performance(pred_terms_list, true_terms_list):
    true_cnt, sys_cnt, hit_cnt = 0, 0, 0
    for pred_terms, true_terms in zip(pred_terms_list, true_terms_list):
        true_cnt += len(true_terms)
        sys_cnt += len(pred_terms)
        hit_cnt += __count_hit(true_terms, pred_terms)

    p, r, f1 = prf1(true_cnt, sys_cnt, hit_cnt)
    return f1


def train(train_X, train_y, valid_X, valid_y, test_X, test_sents, test_token_seqs, true_test_terms_list,
          model, model_fn, optimizer, parameters, epochs=200, batch_size=128, crf=False):
    best_loss = float("inf")
    # valid_history = []
    # train_history = []
    for epoch in range(epochs):
        for batch in batch_generator(train_X, train_y, batch_size, crf=crf):
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask = batch
            loss = model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
        # loss = valid_loss(model, train_X, train_y, crf=crf)
        # train_history.append(loss)
        loss = valid_loss(model, valid_X, valid_y, crf=crf)
        # valid_history.append(loss)
        if loss < best_loss:
            pred_terms_list = test_dhl(model, test_sents, test_X, test_token_seqs, crf=False)
            # cur_f1 = __calc_f1(gold_file, pred_file)
            test_f1 = __get_performance(pred_terms_list, true_test_terms_list)
            print('l_v={:.4f} t_f1={:.4f}'.format(loss, test_f1))
            best_loss = loss
            torch.save(model, model_fn)
            print('model saved to {}'.format(model_fn))
        else:
            print('l_v={:.4f}'.format(loss))
        shuffle_idx = np.random.permutation(len(train_X))
        train_X = train_X[shuffle_idx]
        train_y = train_y[shuffle_idx]
    # model = torch.load(model_fn)
    # return train_history, valid_history


def read_se14_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    sent_texts, terms_list = list(), list()
    for i, sent in enumerate(root.iter("sentence")):
        sent_texts.append(sent.find('text').text)

        terms = list()
        for term_elem in sent.iter('aspectTerm'):
            # print(term_elem.attrib['term'])
            terms.append(term_elem.attrib['term'])
            # terms.append({'term': term_elem.attrib['term'], 'span': (
            #     int(term_elem.attrib['from']), int(term_elem.attrib['to']))})
        terms_list.append(terms)
    return sent_texts, terms_list


def train_ri(gen_emb_file, domain_emb_file, load_model_file, data_file, test_xml_file, test_token_seqs_file,
             output_model_prefix, valid_split, runs, epochs, lr, dropout, batch_size=128):
    #    gen_emb=np.load(data_dir+"gen.vec.npy")
    gen_emb = np.load(gen_emb_file)
    domain_emb = np.load(domain_emb_file)
    print(data_file)
    ae_data = np.load(data_file)
    test_sent_texts, true_test_terms_list = read_se14_xml(test_xml_file)

    with open(test_token_seqs_file) as f:
        test_token_seqs = json.load(f)

    valid_X = ae_data['train_X'][-valid_split:]
    valid_y = ae_data['train_y'][-valid_split:]
    train_X = ae_data['train_X'][:-valid_split]
    train_y = ae_data['train_y'][:-valid_split]

    for r in range(runs):
        print('{}, load model from {}'.format(r, load_model_file))
        if load_model_file is not None:
            model = torch.load(load_model_file)
        else:
            model = Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
        model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        train(train_X, train_y, valid_X, valid_y, ae_data['test_X'], test_sent_texts, test_token_seqs,
              true_test_terms_list, model, output_model_prefix + str(r), optimizer, parameters, epochs, crf=False)


if __name__ == "__main__":
    from platform import platform

    if platform().startswith('Windows'):
        model_dir = 'd:/data/aspect/decnndata/'
        main_data_dir = 'd:/data/aspect'
    else:
        model_dir = '/home/hldai/data/aspect/decnndata/'
        main_data_dir = '/home/hldai/data/aspect'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=model_dir)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--valid', type=int, default=150)  # number of validation data.
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.55)

    args = parser.parse_args()

    domain = 'laptop'
    if args.domain == 'laptop':
        data_file = 'data/prep_data/laptop.npz'
    elif args.domain == 're14':
        data_file = 'data/prep_data/restaurants14-dhl.npz'
        domain = 'restaurant'
    else:
        data_file = 'data/prep_data/restaurants15-dhl.npz'
        domain = 'restaurant'

    gen_emb_file = 'data/prep_data/glove.840B.300d.txt.npy'
    domain_emb_file = 'data/prep_data/laptop_emb.vec.npy'
    load_model_file = os.path.join(main_data_dir, 'decnndata/laptops-decnn.pth')
    # load_model_file = None
    output_model_prefix = os.path.join(main_data_dir, 'decnndata/laptops-decnn-ri')
    # text_file = 'data/prep_data/laptops14-dhl-test-raw.json'
    text_file = 'data/prep_data/laptop_raw_test.json'
    test_gold_file = 'data/official_data/Laptops_Test_Gold.xml'

    train_ri(gen_emb_file, domain_emb_file, load_model_file, data_file, test_gold_file,
             text_file, output_model_prefix, args.valid, args.runs, args.epochs,
             args.lr, args.dropout, args.batch_size)
