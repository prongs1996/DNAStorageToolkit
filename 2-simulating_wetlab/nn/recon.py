'''
Trace reconstruction code
Receive a json file as input,
Perform reconstruction,
Save result to a json file,
And compare with reconstruction result on real data.
Author: Longshen Ou
'''

import os
import sys
import json
import matplotlib.pyplot as plt
from dnacodec.trace_reconstruction.recon_my import *
from evaluate_recon import evaluate
from utils import jpath


def main():
    pass


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


def collate_result(result_root, res_path, recon_ref_path):
    res = read_json(res_path)
    ref = read_json(recon_ref_path)
    res_error = res.pop('pos_error_rate')
    ref_error = ref.pop('pos_error_rate')
    for id in ref:
        res['ref_' + id] = ref[id]
    s = 0
    for i, j in zip(res_error, ref_error):
        s += abs(i - j)
    res['avg_pos_error_rate_dif'] = s / len(ref_error)
    save_json(res, jpath(result_root, 'recon_compare.json'))
    plt.figure(figsize=(4, 3))
    plt.plot(res_error, label='result')
    plt.plot(ref_error, label='reference')
    plt.legend()
    plt.savefig(jpath(result_root, 'recon_compare.png'))


def recon(input_path, coverage, output_path, result_root, ref_data_path, res_save_path, fig_save_path, recon_ref_path):
    '''
    Trace reconstruction
    '''
    data = read_json(input_path)
    clusters = []
    for i in data:
        cluster = []
        if coverage == 5:
            cluster = random.sample(data[i]['syn'], k=coverage)
        elif coverage > 5:
            cluster = data[i]['syn'][:coverage]
        else:
            raise Exception
        clusters.append((i, cluster))

    start = time.time()
    strands = reconstruct_clusters(clusters)
    end = time.time()

    res = {}
    for i, strand in strands:
        res[i] = {'rec': strand}

    save_json(res, output_path)
    print("Trace reconstruction took " + str(end - start) + " seconds")

    evaluate(output_path, ref_data_path, res_save_path, fig_save_path)
    print('Evaluation finished.')

    collate_result(result_root, res_path=res_save_path, recon_ref_path=recon_ref_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DNA consensus')
    parser.add_argument(
        '--i', type=str, default="ClusteredStrands.txt", help="input file")
    parser.add_argument(
        '--o', type=str, default="ReconstructedStrands.txt", help="output file")

    args = parser.parse_args()

    coverage = 5  # cluster that are too large will be shrinked before reconstruction
    data_root = './data/Microsoft_Nanopore'
    result_root = './results/ms_nano/rule_based' + '/../recon_ref'  # generate recon directly on test set
    input_path = jpath(data_root, 'test.json')

    result_root = './results/ms_nano/rule_based'
    input_path = jpath(result_root, 'out.json')
    out_path = jpath(result_root, 'recon_out.json')
    ref_path = jpath(data_root, 'test.json')
    recon_ref_path = './results/ms_nano/recon_ref/recon_result.json'
    res_save_path = jpath(result_root, 'recon_result.json')
    fig_save_path = jpath(result_root, 'recon.png')

    args.i = input_path
    args.o = out_path
    args.coverage = coverage

    recon(args)
