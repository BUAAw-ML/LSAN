import numpy as np
from mxnet.contrib import text
import torch.utils.data as data_utils
import torch
import pickle as pkl

def load_data(batch_size=64):
    X_tst = np.load("../../datasets/ProgrammerWeb/npy_data/test_texts.npy")
    X_trn = np.load("../../datasets/ProgrammerWeb/npy_data/train_texts.npy")
    Y_trn = np.load("../../datasets/ProgrammerWeb/npy_data/train_labels.npy")
    Y_tst = np.load("../../datasets/ProgrammerWeb/npy_data/test_labels.npy")
    # label_embed = np.load("../../datasets/ProgrammerWeb/npy_data/label_embed.npy")
    with open('../../datasets/ProgrammerWeb/word_embedding_model/label_embed.npy', 'rb') as pkl_file:
        label_embed = pkl.load(pkl_file)
    embed = text.embedding.CustomEmbedding('../../datasets/ProgrammerWeb/npy_data/word_embed.txt')
    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    return train_loader, test_loader, label_embed, embed.idx_to_vec.asnumpy(), X_tst, embed.token_to_idx, Y_tst, Y_trn

