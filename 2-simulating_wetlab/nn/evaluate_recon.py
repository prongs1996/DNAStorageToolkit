'''
Evaluate reconstruction accuracy.
Author: Longshen Ou
'''

import matplotlib.pyplot as plt
from utils import read_json, save_json



def evaluate(out_path, ref_path, res_save_path, fig_save_path):
    data = read_out_and_ref(out_path, ref_path)
    for id in data:
        strand_len = len(data[id]['out'])
        break
    error_list = [0 for i in range(strand_len)]
    perfect_cnt = 0
    for id in data:
        out = data[id]['out'].strip()
        ref = data[id]['ref'].strip()
        if out == ref:
            perfect_cnt += 1
        for j in range(strand_len):
            if ref[j] != out[j]:
                error_list[j] += 1
    # print(error_list)
    data_size = len(data)

    # Per position error rate
    error_rate = [i / data_size for i in error_list]

    # Perfect reconstructed strand
    perfect_cnt

    # Average accuracy
    avg_error_rate = sum(error_rate) / strand_len

    res = {
        'avg_error_rate': avg_error_rate,
        'perfect_strand': perfect_cnt,
        'pos_error_rate': error_rate
    }

    save_json(res, res_save_path)
    plt.plot(error_rate)
    plt.savefig(fig_save_path)

def read_out_and_ref(out_path, ref_path):
    '''
    Read reference data and output data,
    save them into a single dict and return
    '''
    out_data = read_json(out_path)
    ref_data = read_json(ref_path)
    if len(ref_data) != len(out_data):
        print('Ref : out = {} : {}'.format(len(ref_data), len(out_data)))
        raise Exception('Unequal output and reference number.')
    data = {}
    for id in out_data:
        data[id] = {'ref': ref_data[id]['ref'], 'out': out_data[id]['rec']}
    return data

if __name__ == '__main__':
    ref_path = './data/test/test.json'
    out_path = './data/test/recon_output.json'
    res_save_path = './results/recon/test.json'
    fig_save_path = './results/recon/test.png'
    evaluate(out_path, ref_path, res_save_path, fig_save_path)