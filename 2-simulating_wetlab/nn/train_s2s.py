'''
Train a sequence-to-sequence neural network
Author: Longshen Ou
'''

import os
import sys
import mlconfig
import random
import time
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
from models.models import *

import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from PIL import Image as Img
from utils import *
from dataset import get_dataset
from recon import recon

from torch.utils.tensorboard import SummaryWriter

random.seed(10)
np.random.seed(0)
torch.manual_seed(10)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def train(hparam, ctn=False):
    model_name = hparam['model_name']
    
    if hparam['use_tensorboard']:
        if not os.path.exists(hparam['output_dir']):
            os.mkdir(hparam['output_dir'])
        else:
            if hparam['debug'] == False:
                raise Exception('Log already exists!')
        hparam.save(jpath(hparam['output_dir'], 'hparam.yaml'))
        writer = SummaryWriter(jpath(hparam['output_dir'], 'tensorboard'))

    dataset_train = get_dataset(hparam['dataset_class'], hparam['train_path'], split='train')
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=hparam['batch_size'],
        shuffle=True,
        num_workers=hparam['num_workers'],
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=dataset_train.collate_fn,
    )

    dataset_valid = get_dataset(hparam['dataset_class'], hparam['valid_path'], split='valid')
    valid_loader = DataLoader(
        dataset=dataset_valid,
        batch_size=hparam['batch_size'],
        shuffle=False,
        num_workers=hparam['num_workers'],
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=dataset_valid.collate_fn,
    )

    dataset_test = get_dataset(hparam['dataset_class'], hparam['test_path'], split='test')
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=hparam['batch_size'],
        shuffle=False,
        num_workers=hparam['num_workers'],
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=dataset_test.collate_fn
    )

    # Model, optimizer, loss function, earlystopping
    net = get_model(model_name, hparam).to(hparam['device'])
    # if ctn == True:
    #     net.load_state_dict(torch.load('models/' + model_name + '.pth'))
    print(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=hparam['LR'], weight_decay=hparam['WEIGHT_DECAY'])
    loss_func = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(hparam['PATIENCE'], verbose=False, savefolder=hparam['output_dir'])

    param_num = check_model(net)
    
    with open(jpath(hparam['output_dir'], 'net_params.txt'), 'w') as f:
        f.write('{}\n'.format(net))
        f.write('Num of param: {:,}'.format(param_num))

    if hparam['check_only'] == True:
        return

    # Training and validating
    time_begin = time.time()
    for epoch in range(hparam['EPOCH']):
        net.train()
        running_loss = 0.0
        
        if epoch > 0:
            pbar = tqdm(train_loader)
            for step, batch in enumerate(pbar):
                enc_inp, dec_inp, label = [i.to(hparam['device']) for i in batch]
                out, attn = net(enc_inp, dec_inp)
                out = out.permute(0, 2, 1)
                loss = loss_func(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_description('Epoch: {} | Step {} / {} | loss {:.4f}'.format(epoch, step + 1, len(train_loader), loss.item()))

            if epoch % hparam['REDUCE_EPOCH'] == 0 and epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= hparam['REDUCE_FACTOR']
        else:
            running_loss = 1e9

        avg_train_loss = running_loss / len(train_loader)
        print('Epoch: {} | Training Loss: {:.4f} '.format(epoch, avg_train_loss), end='')

        # Validating
        running_loss = 0.0
        net.eval()
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                enc_inp, dec_inp, label = [i.to(hparam['device']) for i in batch]
                out, attn = net(enc_inp, dec_inp)
                out = out.permute(0, 2, 1)
                loss = loss_func(out, label)
                running_loss += loss.item()
        avg_valid_loss = running_loss / len(valid_loader)
        print(
            ' | Validation Loss: {:.4f} '.format(avg_valid_loss), end='')

        # Validating by test data
        running_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                enc_inp, dec_inp, label = [i.to(hparam['device']) for i in batch]
                out, attn = net(enc_inp, dec_inp)
                out = out.permute(0, 2, 1)
                loss = loss_func(out, label)
                running_loss += loss.item()
        avg_test_loss = running_loss / len(test_loader)
        print(' | Test Loss: {:.4f} '.format(avg_test_loss))

        # Visualization
        if hparam['use_tensorboard']:
            writer.add_scalars('Training Loss Graph', {'train_loss': avg_train_loss,
                                                       'validation_loss': avg_valid_loss,
                                                       'test_loss': avg_test_loss}, epoch)

        log_str = 'Epoch: {} | Training Loss: {:.4f} | Validation Loss: {:.4f} | Test Loss: {:.4f}\n'.format(
            epoch, avg_train_loss, avg_valid_loss, avg_test_loss)
        with open(hparam['log_fn'], 'a') as f:
            f.write(log_str)

        time.sleep(0.5)

        if hparam['EARLY_STOP']:
            early_stopping(avg_valid_loss, net, epoch)
            if early_stopping.early_stop == True:
                print("Early Stopping!")
                with open(hparam['log_fn'], 'a') as f:
                    f.write('Early stop at epoch {}. Best model from epoch {}'.format(epoch, early_stopping.best_epoch))
                break



def infer(hparam):
    '''
    Generate noisy samples using the trained model.
    '''
    model_path = jpath(hparam['output_dir'], 'checkpoint.pth')
    net = get_model(hparam['model_name'], hparam).to(hparam['device'])
    net.load_state_dict(torch.load(model_path))

    dataset_test = get_dataset(hparam['dataset_class'], hparam['test_path'], split='test')
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=hparam['batch_size'],
        shuffle=False,
        num_workers=hparam['num_workers'],
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=dataset_test.collate_fn,
    )
    ret = {}
    net.eval()
    bs = hparam['generate_bs']
    with torch.no_grad():
        data = read_json(hparam['test_path'])
        data = list(data.items())
        for i in tqdm(range(0, len(data), bs)):
            i_end = i + bs
            clean = [j[1]['ref'] for j in data[i:i_end]]
            enc_inp = dataset_test.tokenizer.batch_tokenize(clean)
            enc_inp = torch.tensor(enc_inp).to(hparam['device']) # [B, L]
            enc_out, _ = net.encoder(enc_inp)       # [B, L_in, D]

            syn = []
            for j in range(hparam['generation_coverage']):
                net.decoder.attn.reset()
                c = torch.zeros(
                    enc_out.shape[0], net.decoder.attn_dim, device=enc_out.device
                )
                hs = None
                enc_len = torch.tensor([enc_out.shape[1]]).long()
                input_ids = torch.ones(size=[enc_out.shape[0], 1], dtype=torch.long).to(hparam['device']) # [B, 1], <bos> for all seq
                # input_ids = torch.tensor([[1]]).to(hparam['device'])
                
                outputs_lst, attn_lst = [], []
                for k in range(130-1):
                    input_emb = net.decoder.emb(input_ids) # [B=1, L, D]
                    outputs, hs, c, w = net.decoder.forward_step(
                        input_emb[:, k], hs, c, enc_out, enc_len
                    )
                    logits = net.lm_head(outputs) # [B, V]
                    outputs_lst.append(outputs)
                    attn_lst.append(w)
                    
                    # greedy sampling
                    ch = logits_to_ch(logits).unsqueeze(1).to(hparam['device']) # [B,1]

                    input_ids = torch.cat((input_ids, ch), dim=1)
                out = input_ids
                syn.append(out)
            syn = torch.stack(syn) # syn: [coverage, batch, len]
            syn = syn.permute(1, 0, 2) # [batch, coverage, len]
            
            for entry, ref, noisy in zip(data[i:i_end], clean, syn):
                id = entry[0]
                noisy = [dataset_test.tokenizer.de_tokenize(i) for i in noisy.tolist()]
                ret[id] = {
                    'ref':ref,
                    'syn':noisy,
                }
            print(len(ret))
    save_json(ret, jpath(hparam['output_dir'], 'synthesized.json'))

    

def evaluate(hparam):
    '''
    Perform reconstruction on the generated samples, and compare with reference.
    '''
    recon(
        input_path=jpath(hparam['output_dir'], 'synthesized.json'),
        coverage=hparam['generation_coverage'],
        output_path=jpath(hparam['output_dir'], 'recon_out.json'),
        result_root=hparam['output_dir'],
        ref_data_path=jpath(hparam['dataset_root'], 'test.json'),
        res_save_path=jpath(hparam['output_dir'], 'recon_result.json'),
        fig_save_path=jpath(hparam['output_dir'], 'recon.png'),
        recon_ref_path=hparam['recon_ref_path']
    )


def pack_batch(batch, hparam):
    b_clean1 = batch[:,0,:]
    b_clean2 = batch[:,1,:]
    b_noisy1 = batch[:,2,:]
    b_noisy2 = batch[:,3,:]
    b_clean = torch.cat((b_clean1, b_clean2), dim=0)
    b_noisy = torch.cat((b_noisy1, b_noisy2), dim=0)

    b_x = b_clean.to(hparam['device'])
    b_y = b_noisy.to(hparam['device'])
    return b_x, b_y

def logits_to_ch(logits):
    '''
    Do the sampling for logits, generate character sequence
    logits: [bs, vocab_size]
    '''
    ret = torch.zeros(size=(logits.shape[0],), dtype=torch.long) # [bs*2, len/2]
    t = torch.softmax(logits, dim=1) # [bs, vocab_size]
    for i in range(ret.shape[0]):
        dist = t[i]
        # print(dist)
        ch = np.random.choice(np.arange(len(dist)), p=dist.cpu().numpy())
        # print(dist.shape, ch)
        ret[i] = ch
            # exit(20)
    
    # t = torch.softmax(logits, dim=2)
    # print(t[:, -5:])
    # ret = t.argmax(axis=2)
    # print(ret.shape) # [bs*2, len/2]
    # print(ret[:, -5:])
    return ret

def logits_to_seq(logits):
    '''
    logits: [bs*2, len/2, vocab_size]
    '''
    ret = torch.zeros(size=(logits.shape[0], logits.shape[1]), dtype=torch.long) # [bs*2, len/2]
    t = torch.softmax(logits, dim=2) # [bs*2, len/2, vocab_size]
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            dist = t[i][j]
            ch = np.random.choice(np.arange(5), p=dist.cpu().numpy())
            # print(dist.shape, ch)
            ret[i][j] = ch
            # exit(20)
    
    # t = torch.softmax(logits, dim=2)
    # print(t[:, -5:])
    # ret = t.argmax(axis=2)
    # print(ret.shape) # [bs*2, len/2]
    # print(ret[:, -5:])
    return ret

def post_process(out, bs):
    '''
    Resume a batch of data back to full length
    Ensure the characters are legal
    out: a batch of output
    '''
    # assert out.shape[0] // 2 == bs
    assert out.shape[0] % 2 == 0
    half = out.shape[0] // 2

    vocab_dict = {
        1: 'A',
        2: 'T',
        3: 'C',
        4: 'G',
    }

    out_1, out_2 = out[:half], out[half:]
    # print(out_1.shape, out_2.shape)
    ret = []
    for i in range(half):
        t1 = depad(out_1[i].tolist())
        t2 = depad(out_2[i].tolist())
        t = t1[::-1] + t2

        # Convert ids to characters
        # While ensure there is no '0' inside the sequence
        res = []
        for j in t:
            if j != 0:
                res.append(vocab_dict[j])
            else:
                res.append(vocab_dict[random.choice([1,2,3,4])])
        res = ''.join(res)

        ret.append(res)
    return ret

def depad(seq):
    '''
    Remove 0 at the end of sequence
    seq: a list of numbers
    '''
    while seq[-1] == 0:
        seq.pop(-1)
    return seq

def compute_pred(out):
    pred_y = out.detach()
    pred_y[pred_y >= 0.5] = 1
    pred_y[pred_y < 0.5] = 0
    pred_y = pred_y.data.cpu().numpy()
    return pred_y

def check_model(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Totalparams:', format(pytorch_total_params, ','))
    print('Trainableparams:', format(pytorch_train_params, ','))
    return pytorch_train_params


if __name__ == '__main__':
    arg_path = sys.argv[1]  # e.g., './hparams/iterative_bt/s2t.yaml'
    args = mlconfig.load(arg_path)
    hparam = args
    os.environ["CUDA_VISIBLE_DEVICES"] = hparam['gpu']
    # train(hparam)
    # infer(hparam)
    evaluate(hparam)