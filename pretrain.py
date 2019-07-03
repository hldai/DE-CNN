import argparse
import torch
import numpy as np
import random
import datautils
from decnn import batch_generator, Model

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


def get_data(tok_texts_file, tokfc_texts_file, terms_file, token_id_file):
    terms_true_list = datautils.load_json_objs(terms_file)
    f_tok = open(tok_texts_file, encoding='utf-8')
    f_tokfc = open(tokfc_texts_file, encoding='utf-8')
    for i, (line_tok, line_tokfc) in enumerate(zip(f_tok, f_tokfc)):
        if i > 10:
            break
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


def __pretrain(tok_texts_file, tokfc_texts_file, terms_file, token_id_file, valid_ids_file,
               n_epochs, batch_size, use_crf=False):
    get_data(tok_texts_file, tokfc_texts_file, terms_file, token_id_file)

    best_loss = float("inf")
    for epoch in range(n_epochs):
        for batch in batch_generator(train_X, train_y, batch_size, crf=use_crf):
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask = batch
            loss = model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
        loss = valid_loss(model, train_X, train_y, crf=use_crf)
        loss = valid_loss(model, valid_X, valid_y, crf=use_crf)
        print('l_v={:.4f}'.format(loss))
        if loss < best_loss:
            best_loss = loss
            torch.save(model, model_fn)
            print('model saved to {}'.format(model_fn))
        shuffle_idx = np.random.permutation(len(train_X))
        train_X = train_X[shuffle_idx]
        train_y = train_y[shuffle_idx]


def run(domain, data_file, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, batch_size=128):
    #    gen_emb=np.load(data_dir+"gen.vec.npy")
    gen_emb = np.load(data_dir + "glove.840B.300d.txt.npy")
    domain_emb = np.load(data_dir + domain + "_emb.vec.npy")
    print(data_file)
    ae_data = np.load(data_file)

    valid_X = ae_data['train_X'][-valid_split:]
    valid_y = ae_data['train_y'][-valid_split:]
    train_X = ae_data['train_X'][:-valid_split]
    train_y = ae_data['train_y'][:-valid_split]

    for r in range(runs):
        print(r)
        model = Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
        model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        train_history, valid_history = train(train_X, train_y, valid_X, valid_y, model, model_dir + domain + str(r),
                                             optimizer, parameters, epochs, crf=False)


if __name__ == "__main__":
    from platform import platform

    if platform().startswith('Windows'):
        model_dir = 'd:/data/aspect/decnn-models/'
    else:
        model_dir = '/home/hldai/data/aspect/decnn-models/'

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

    tok_texts_file = ''
    tokfc_texts_file = ''
    terms_file = ''
    token_id_file = ''
    valid_ids_file = ''
    n_epochs = 35
    batch_size = 32
    __pretrain(tok_texts_file, tokfc_texts_file, terms_file, token_id_file, valid_ids_file, n_epochs, batch_size)
