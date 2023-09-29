'''
Dataset class and validation scripts for training.
Author: Longshen Ou
'''

import os
import math
import torch
import random
import pandas as pd
from utils import jpath, read_json, S2sTokenizer, print_json
from torch.utils.data import Dataset, DataLoader

def main():
    check_s2s()

def check_s2s():
    dataset = S2sDataset('./data/Microsoft_Nanopore/train.json', 'valid')
    print('Dataset size:', len(dataset))
    train_loader = DataLoader(
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        batch_size=3,
        shuffle=True,
        num_workers=0,
    )
    for step, batch in enumerate(train_loader):
        # print(batch)
        b_x, b_y, b_z = batch
        # for x,y in zip(b_x, b_y):
        #     print(x)
        #     print(y)
        print(b_x.shape, b_y.shape, b_z.shape)
        print(b_x[0])
        print(b_y[0])
        print(b_z[0])
        # b_clean = torch.cat((b_clean1, b_clean2), dim=0)
        # b_noisy = torch.cat((b_noisy1, b_noisy2), dim=0)
        # print(b_clean.shape)
        break

def check_my():
    dataset = S2sDataset('./data/Microsoft_Nanopore/train.json', 'valid')
    print(len(dataset))
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
    )
    for step, batch in enumerate(train_loader):
        print(batch)
        b_clean1 = batch[:,0,:]
        b_clean2 = batch[:,1,:]
        b_noisy1 = batch[:,2,:]
        b_noisy2 = batch[:,3,:]
        b_clean = torch.cat((b_clean1, b_clean2), dim=0)
        b_noisy = torch.cat((b_noisy1, b_noisy2), dim=0)
        print(b_clean.shape)
        break

def get_dataset(dataset_class, data_path, split):
    C = eval(dataset_class)
    dataset = C(data_path, split)
    return dataset

class S2sDataset(Dataset):
    '''
    Dataset class for sequence-to-sequence model
    '''
    def __init__(self, data_path, split='train'):
        '''
        data_path: path to the dataset file, should be a json file like
        {
            id: {
            'ref': ref seq,
            'syn': [
                noisy seq 1,
                noisy seq 2,
                ...
            ]
            }
        }
        split: dataset split
        '''
        assert split in ['train', 'valid', 'test']

        self.data_root = os.path.dirname(data_path)
        self.split = split
        data = read_json(data_path)
        self.data = []
        for id in data:
            for s in data[id]['syn']:
                self.data.append((data[id]['ref'], s))
        self.tokenizer = S2sTokenizer()
        self.vocab_size = 8
        max_strand_len = 0
        for i in self.data:
            max_strand_len = max(max_strand_len, len(i[1]))
        self.max_strand_len = max_strand_len
        # print('Max strand length:', self.max_strand_len)
        self.max_half_len = math.ceil(self.max_strand_len/2.0)

    def __getitem__(self, idx):
        '''
        Return the clean strand with the corresponding index,
        together with a noisy strand random chosen from 'syn'
        '''
        clean, noisy = self.data[idx]
        return [clean, noisy]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, l):
        b_clean = [i[0] for i in l]
        b_noisy = [i[1] for i in l]
        b_clean_ids = torch.tensor(self.tokenizer.batch_tokenize(b_clean))
        b_noisy_ids = torch.tensor(self.tokenizer.batch_tokenize(b_noisy))
        enc_inp = b_clean_ids
        dec_inp = b_noisy_ids
        label = torch.zeros_like(dec_inp)
        label[:, :-1] = dec_inp[:, 1:]
        
        return enc_inp, dec_inp, label

class MyDataset(Dataset):
    '''
    Dataset class for MLP
    '''
    def __init__(self, data_path, split='train'):
        '''
        data_path: path to the dataset file, should be a json file like
        {
            id: {
            'ref': ref seq,
            'syn': [
                noisy seq 1,
                noisy seq 2,
                ...
            ]
            }
        }
        split: dataset split
        '''
        assert split in ['train', 'valid', 'test']

        self.data_root = os.path.dirname(data_path)
        self.split = split
        data = read_json(data_path)
        self.data = []
        for id in data:
            for s in data[id]['syn']:
                self.data.append((data[id]['ref'], s))
        self.dict = {
            'A':1, # 0 is padding
            'T':2,
            'C':3,
            'G':4
        }
        max_strand_len = 0
        for i in self.data:
            max_strand_len = max(max_strand_len, len(i[1]))
        self.max_strand_len = max_strand_len
        # print('Max strand length:', self.max_strand_len)
        self.max_half_len = math.ceil(self.max_strand_len/2.0)

    def __getitem__(self, idx):
        '''
        Return the clean strand with the corresponding index,
        together with a noisy strand random chosen from 'syn'
        '''
        entry = self.data[idx]
        
        # Clean strands: convert to tensor directly
        clean = [self.dict[i] for i in entry[0]]
        noisy = [self.dict[i] for i in entry[1]]

        # Cut the two strands in the middle
        len_clean_half = len(clean) // 2
        clean_1 = clean[:len_clean_half]
        clean_2 = clean[len_clean_half:][::-1]

        len_noisy_half = len(noisy) // 2
        noisy_1 = noisy[:len_noisy_half]
        noisy_2 = noisy[len_noisy_half:][::-1]

        # Pad and create tensor
        for strand in [clean_1, clean_2, noisy_1, noisy_2]:
            pad_size = self.max_half_len - len(strand)
            strand += [0 for i in range(pad_size)]
        # print([clean_1, clean_2, noisy_1, noisy_2])

        return torch.tensor([clean_1, clean_2, noisy_1, noisy_2])

    def tokenize(self, seq):
        '''
        Convert a sequence of character (string) to 
        TWO lists of number
        '''
        t = [self.dict[i] for i in seq]
        len_half = len(t) // 2
        first_half = t[:len_half]
        second_half = t[len_half:][::-1]
        for strand in [first_half, second_half]:
            pad_size = self.max_half_len - len(strand)
            strand += [0 for i in range(pad_size)]
        return first_half, second_half

    def __len__(self):
        return len(self.data)

  

if __name__ == '__main__':
    main()