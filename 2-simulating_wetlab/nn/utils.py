'''
Tokenizer, earlystopping, and other util functions
Author: Longshen Ou
'''

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

ls = os.listdir
jpath = os.path.join


def main():
    check_tokenizer()


def check_tokenizer():
    seq = 'ATCNG'
    tk = S2sTokenizer()
    ids = tk.tokenize(seq)
    print(ids)
    ids = tk.pad(ids, 10)
    print(ids)
    s = tk.de_tokenize(ids)
    print(s)

    b = [
        'ATCG',
        'GGCTACCC',
    ]
    ids = tk.batch_tokenize(b)
    print(ids)

    for id in ids:
        print(tk.de_tokenize(id))


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        data = json.loads(data)
    return data


def save_json(data, path, sort=False):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, sort_keys=sort, ensure_ascii=False))


def print_json(data):
    print(json.dumps(data, indent=4, ensure_ascii=False))


def timecode_to_timedelta(timecode):
    '''
    Convert timecode 'MM:SS.XXX' to timedelta object
    '''
    m, s = timecode.strip().split(':')
    m = int(m)
    s = float(s)
    ret = timedelta(minutes=m, seconds=s)
    return ret


def sec_to_timedelta(time_in_sec):
    '''
    Convert 'sec.milli' to timedelta object
    :param time_in_sec: string, time in second
    '''
    time_in_sec = float(time_in_sec)
    ret = timedelta(seconds=time_in_sec)
    return ret


def update_dic(d, v):
    if v in d:
        d[v] += 1
    else:
        d[v] = 1


def sort_dic(d):
    t = list(d.items())
    t.sort()
    ret = dict(t)
    return ret


def plot_dic(d):
    items = list(d.items())
    x = [i[0] for i in items]
    y = [i[1] for i in items]
    plt.plot(x, y)
    plt.show()


def plot_dic_hist(d):
    # Extracting values and their corresponding frequencies
    values = list(d.keys())
    frequencies = list(d.values())

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(values, frequencies, color='blue', alpha=0.7)

    # Adding title and labels
    plt.title('Histogram from Dictionary')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()


class S2sTokenizer:
    def __init__(self):
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        self.dic = {
            'A': 4,
            'T': 5,
            'C': 6,
            'G': 7
        }
        self.dic_rev = {
            0: 'N',
            1: 'N',
            2: 'N',
            3: 'N',
            4: 'A',
            5: 'T',
            6: 'C',
            7: 'G',
        }

    def tokenize(self, seq):
        '''
        Convert a string of characters to a list of ids
        '''
        ids = [self.dic[i] if i in self.dic else self.unk_token for i in seq]
        ids.insert(0, self.bos_token)
        ids.append(self.eos_token)
        return ids

    def pad(self, l, max_len):
        '''
        Pad a tokenized sequence to target length
        '''
        ret = l + [0 for i in range(max_len - len(l))]
        return ret

    def batch_tokenize(self, batch):
        max_len = max([len(i) for i in batch]) + 2
        ret = [self.pad(self.tokenize(seq), max_len) for seq in batch]
        return ret

    def de_tokenize(self, l, min_len=None):
        '''
        Convert a list of ids back to a string
        '''
        # Delete bos token
        while len(l) > 0 and l[0] == self.bos_token:
            l = l[1:]
        # Find the first eos or pad token
        eos_index = -1 if min_len == None else min_len
        for i, v in enumerate(l):
            if min_len != None and i < min_len:
                continue
            if v in [self.eos_token, self.pad_token]:
                eos_index = i
                break
        l = l[:eos_index]

        t = [self.dic_rev[i] for i in l]
        ret = ''.join(t)
        return ret


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, savefolder=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.savefolder = savefolder
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.savefolder == None:
            torch.save(model.state_dict(), 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), os.path.join(self.savefolder, 'checkpoint.pth'))

        self.val_loss_min = val_loss


if __name__ == '__main__':
    main()
