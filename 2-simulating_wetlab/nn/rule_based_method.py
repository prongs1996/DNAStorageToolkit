'''
A rule-based method to synthesize noisy strands
Author: Longshen Ou
'''
import os
import numpy as np
from tqdm import tqdm
from utils import jpath, read_json, save_json
from recon import recon

data_root = './data/Microsoft_Nanopore'
coverage = 5 # coverage when generating noisy strands
out_dir = './results/ms_nano/rule_based'
out_path = jpath(out_dir, 'out.json') # path to save generation result
recon_ref_path = './results/ms_nano/recon_ref/recon_result.json'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def main():
    procedures()


def procedures():
    validate_reference_length()
    construct_error_matrix(jpath(data_root, 'train.json'))
    construct_error_matrix(jpath(data_root, 'valid.json'))
    construct_error_matrix(jpath(data_root, 'test.json'))
    visulize_error_mats()
    visulize_error_mats2()

    generate_noisy_strands()
    evaluate()

def evaluate():
    '''
    Perform reconstruction on the generated samples, and compare with reference.
    '''
    recon(
        input_path=out_path,
        coverage=coverage,
        output_path=jpath(out_dir, 'recon_out.json'),
        result_root=out_dir,
        ref_data_path=jpath(data_root, 'test.json'),
        res_save_path=jpath(out_dir, 'recon_result.json'),
        fig_save_path=jpath(out_dir, 'recon.png'),
        recon_ref_path=recon_ref_path
    )

def generate_noisy_strands():
    '''
    Generate noisy strands on test set according to error mat on training set.
    Generate #coverage times
    '''
    error_mat = read_json(jpath(out_dir, 'error_mat_train.json'))
    data = read_json(jpath(data_root, 'test.json'))
    generated = {}
    for id in tqdm(data):
        clean = data[id]['ref']
        syn = []
        for i in range(coverage):
            noisy = generate(clean, error_mat)
            syn.append(noisy)
        generated[id] = {'ref': clean, 'syn': syn}
    save_json(generated, out_path)        
        

def generate(x, mat):
    '''
    Generate noisy strand given a clean strand and a error mat
    '''
    error_dic = {  # 12 types of errors         error_dic[from][to] = error_id
        'A': {
            'T': 0,
            'C': 1,
            'G': 2,
        },
        'T': {
            'A': 3,
            'C': 4,
            'G': 5,
        },
        'C': {
            'A': 6,
            'T': 7,
            'G': 8,
        },
        'G': {
            'A': 9,
            'T': 10,
            'C': 11,
        },
    }
    ret = []
    # for each position
    for i in range(len(x)):
        all_chs = ['A', 'T', 'C', 'G']
        t = all_chs.copy()
        cur_ch = x[i]  # current character
        other_chs = t.remove(cur_ch)
        prob = {}  # The probability of current position to be different characters
        s = 0
        for ch in t:
            error_type = error_dic[x[i]][ch]
            prob[ch] = mat[i][error_type]
            s += prob[ch]
        prob[cur_ch] = 1 - s

        options = []
        prob_list = []
        for id in prob:
            options.append(id)
            prob_list.append(prob[id])

        # print(options, prob_list)
        index = np.random.choice(np.arange(4), p=prob_list)
        choice = options[index]
        
        ret.append(choice)
    ret = ''.join(ret)
    return ret


def visulize_error_mats():
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 12, figsize=(15, 10))

    splits = ['train', 'valid', 'test']
    for i in range(3):  # for different split
        split = splits[i]
        error_mat = np.array(
            read_json(jpath(out_dir, 'error_mat_{}.json'.format(split))))
        for j in range(12):  # for different error type
            data = error_mat[:, j]
            axs[i, j].plot(data)

    for ax in axs.flat:
        ax.set(xlabel='position', ylabel='probability', ylim=[0, 0.05])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(jpath(out_dir, 'error_mat.png'))

def visulize_error_mats2():
    '''
    Visualize error mat with heatmap
    '''
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 5))

    splits = ['train', 'valid', 'test']
    for i in range(3):  # for different split
        split = splits[i]
        error_mat = np.array(read_json(jpath(out_dir, 'error_mat_{}.json'.format(split)))).T
        t = axs[i].matshow(error_mat)
        axs[i].set_title(split)
        fig.colorbar(t)

    # data = read_json(jpath(out_dir, 'error_mat_{}.json'.format('train')))
    # data = np.array(data).T
    # print(type(data), len(data))
    # plt.matshow(data)
    plt.savefig(jpath(out_dir, 'error_mat_2d.png'))

def validate_reference_length():
    data_root = './data/Microsoft_Nanopore'
    for split in ['train', 'valid', 'test']:
        data_path = jpath(data_root, split+'.json')
        data = read_json(data_path)
        for id in data:
            lent = len(data[id]['ref'])
            assert lent == 110
    print('Strand length validation passed')


def construct_error_matrix(data_path):
    split = data_path.split('/')[-1].split('.')[0]
    strand_len = 110
    half_len = 110 // 2
    error_dic = {  # 12 types of errors         error_dic[from][to] = error_id
        'A': {
            'T': 0,
            'C': 1,
            'G': 2,
        },
        'T': {
            'A': 3,
            'C': 4,
            'G': 5,
        },
        'C': {
            'A': 6,
            'T': 7,
            'G': 8,
        },
        'G': {
            'A': 9,
            'T': 10,
            'C': 11,
        },
    }
    error_cnt = np.zeros([strand_len, 12])
    data_train = read_json(data_path)
    for id in data_train:
        entry = data_train[id]
        ref = entry['ref']
        noisy_strands = entry['syn']

        # Count error type from left to right
        for i in range(0, half_len):
            for strand in noisy_strands:
                if ref[i] != strand[i]:
                    error_type = error_dic[ref[i]][strand[i]]
                    error_cnt[i][error_type] += 1

        # Count error type from right to left
        for i in range(0, half_len):
            pos = -1 - i
            for strand in noisy_strands:
                if ref[pos] != strand[pos]:
                    error_type = error_dic[ref[pos]][strand[pos]]
                    error_cnt[pos][error_type] += 1

    read_cnt = 0
    for id in data_train:
        entry = data_train[id]
        read_cnt += len(entry['syn'])
    error_mat = error_cnt / read_cnt

    save_json(error_mat.tolist(), jpath(
        out_dir, 'error_mat_{}.json'.format(split)))


if __name__ == '__main__':
    main()
