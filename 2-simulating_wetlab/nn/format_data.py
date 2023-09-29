'''
Convert raw data to the format (json) that can be received by trace reconstruction script.
Author: Longshen Ou
'''

import os
import sys
import random
from tqdm import tqdm
from utils import save_json, jpath, read_json


def main():
    format_our_data()
    pass


def procedures():
    format_ms_nano()
    data_statistics('./data/Microsoft_Nanopore/full.json')
    split_dataset('./data/Microsoft_Nanopore/full.json')
    split_statistics('./data/Microsoft_Nanopore/full.json')
    validate_split_ms_nano('./data/Microsoft_Nanopore')

    # fastq_parser('./data/Our_Illumina/Alice_All_3/Alice_All_3_DKDL210005527-1a_HCK5MDSX2_L2_1.fq')
    # fastq_parser('./data/Our_Illumina/Alice_All_3/Alice_All_3_DKDL210005527-1a_HCK5MDSX2_L2_2.fq')

    # fastq_parser('./data/Stanford_Nanopore/raw/merged.fastq')
    # format_ms_stanford_nano()

    # fastq_parser('./data/Microsoft_Illumina/raw/id20.fastq')
    # format_ms_illumina()


def format_our_data():
    data_dir = './data/Ours/'
    data_fp = jpath(data_dir, 'EncodedStrands.txt')
    with open(data_fp) as f:
        data = f.readlines()

    # Write in json
    res = {}
    id = 0
    for line in data:
        line = line.strip()
        if len(line) == 0:
            continue
        res[id] = {'ref': line}
        id += 1
    out_fp = jpath(data_dir, 'clean.json')
    save_json(res, out_fp)


def validate_split_ms_nano(data_root):
    '''
    Ensure every strand in valid or sample split at least has 5 noisy copies.
    '''
    for split in ['valid', 'test']:
        data_path = jpath(data_root, '{}.json'.format(split))
        data = read_json(data_path)
        for id in data:
            assert len(data[id]['syn']) >= 5


def split_statistics(data_path):
    train_data = read_json(data_path.replace('full.json', 'train.json'))
    valid_data = read_json(data_path.replace('full.json', 'valid.json'))
    test_data = read_json(data_path.replace('full.json', 'test.json'))
    print('Train:Valid:Test = {:,} : {:,} : {:,}'.format(
        len(train_data), len(valid_data), len(test_data)))


def split_dataset(data_path):
    '''
    Split the full dataset into train, validation, and test split
    Ratio: train:valid:test = 8:1:1
    Before splitting, all starnds with empty cluster are removed
    '''
    data = read_json(data_path)
    data_old = data
    data = {}
    for id in data_old:
        if len(data_old[id]['syn']) > 0:
            data[id] = data_old[id]
    print('{} non-empty clusters'.format(len(data)))

    # Split
    test_size = int(len(data) / 10)
    data_entries = list(data.items())
    random.shuffle(data_entries)
    train_data = []
    valid_data = []
    test_data = []
    valid_full = False
    test_full = False
    for entry in data_entries:
        # print(len(entry[1]['syn']))
        if valid_full == False:
            if len(entry[1]['syn']) > 5:
                valid_data.append(entry)
                if len(valid_data) == test_size:
                    valid_full = True
            else:
                train_data.append(entry)
        elif test_full == False:
            if len(entry[1]['syn']) > 5:
                test_data.append(entry)
                if len(test_data) == test_size:
                    test_full = True
            else:
                train_data.append(entry)
        else:
            train_data.append(entry)

    train_data = dict(train_data)
    valid_data = dict(valid_data)
    test_data = dict(test_data)

    save_json(train_data, data_path.replace('full.json', 'train.json'))
    save_json(valid_data, data_path.replace('full.json', 'valid.json'))
    save_json(test_data, data_path.replace('full.json', 'test.json'))


def format_ms_stanford_nano():
    with open('./data/Stanford_Nanopore/raw/merged.fa') as f:
        data = f.readlines()
        data = [i.strip() for i in data]
    l = 0
    cnt = 0
    strand_len = {}
    for line in data:
        line_len = len(line)
        l = max(l, line_len)
        cnt += line_len
        if line_len in strand_len:
            strand_len[line_len] += 1
        else:
            strand_len[line_len] = 1
    print(l, cnt / len(data))
    save_json(strand_len, 'strand_length_dist1.json', sort=True)


def format_ms_illumina():
    data_folder = './data/Microsoft_Illumina'
    strand_path = jpath(data_folder, 'raw/id20.refs.txt')
    cluster_path = jpath(data_folder, 'raw/1m.txt')
    # print(lines[:10])
    # data = read_json(cluster_path)
    # print(len(data))

    with open(cluster_path) as f:
        data = f.readlines()
    l = 0
    cnt = 0
    strand_len = {}
    for line in data:
        line_len = len(line)
        l = max(l, line_len)
        cnt += line_len
        if line_len in strand_len:
            strand_len[line_len] += 1
        else:
            strand_len[line_len] = 1
    print(l, cnt / len(data))
    save_json(strand_len, 'strand_length_dist1.json', sort=True)


def fastq_parser(fn):
    '''
    Code from https://www.biostars.org/p/317524/
    Parse fastq file to a 'NoisyStrands.txt' file for clustering
    '''

    def process(lines=None):
        ks = ['name', 'sequence', 'optional', 'quality']
        ret = {k: v for k, v in zip(ks, lines)}
        del ret['optional']
        del ret['quality']
        return ret

    if not os.path.exists(fn):
        raise SystemError("Error: File does not exist\n")

    n = 4
    with open(fn, 'r') as fh:
        data = fh.readlines()

    records = []
    lines = []
    output_path = fn + '.txt'
    with open(output_path, 'w') as f:
        for line in tqdm(data):
            lines.append(line.rstrip())
            if len(lines) == n:
                record = process(lines)
                # sys.stderr.write("Record: %s\n" % (str(record)))
                # records.append(record)
                lines = []
                f.write(record['sequence'] + '\n')
    # save_json(records, fn+'.json')


def format_ms_nano():
    data_folder = './data/Microsoft_Nanopore'
    clean_path = jpath(data_folder, 'raw/Centers.txt')
    cluster_path = jpath(data_folder, 'raw/Clusters.txt')

    with open(clean_path) as f:
        strands = f.readlines()
        strands = [i.strip() for i in strands]
    with open(cluster_path) as f:
        clusters_raw = f.readlines()
        clusters_raw = [i.strip() for i in clusters_raw]

    cluster_cnt = 0
    clusters = []
    currrent_cluster = None  # Note: there are some empty clusters
    for line in clusters_raw:
        if line[0] == '=':
            cluster_cnt += 1
            if currrent_cluster != None:
                clusters.append(currrent_cluster)
            currrent_cluster = []
        else:
            currrent_cluster.append(line)
    clusters.append(currrent_cluster)

    # Write in json
    res = {}
    if len(strands) != len(clusters):
        print(len(strands), len(clusters))
        raise Exception('Incorrect cluster num')

    for i in range(len(clusters)):
        res[i + 1] = {'ref': strands[i], 'syn': clusters[i]}
    save_json(res, jpath(data_folder, 'full.json'))


def data_statistics(data_path):
    data = read_json(data_path)
    cluster_num = len(data)
    min_size, max_size = 100, 0
    cnt = 0
    for id in data:
        cluster_size = len(data[id]['syn'])
        cnt += cluster_size
        max_size = max(max_size, cluster_size)
        min_size = min(min_size, cluster_size)
    avg_size = cnt / cluster_num
    print('Total cluster num: ', cluster_num)
    print('Avg, Min, Max cluster size: ', avg_size, min_size, max_size)


if __name__ == '__main__':
    main()
