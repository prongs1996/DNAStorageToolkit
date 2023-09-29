'''
Check edit distance for between clean and generated strands
'''

import os
import sys
import editdistance

from utils import *


def _main():
    check_edit_distances()
    # check_strand_length()
    # check_output_length()
    # check_strand_length_clean()


def check_edit_distances():
    # # Check nanopore test split original
    # original_fp = './data/Microsoft_Nanopore/test.json'
    # check_edit_distance(original_fp)
    #
    # # Check generated strands for test split
    # output_dir = './results/test_edit_distance'
    # output_fp = jpath(output_dir, 'synthesized.json')
    # check_edit_distance(output_fp)

    # # Check generated strands using new detokenize code
    # output_dir = './results/test_edit_distance_new_dec'
    # output_fp = jpath(output_dir, 'synthesized.json')
    # check_edit_distance(output_fp)

    fp = './results/test_edit_distance_ours_1000_+-10/synthesized.json'
    check_edit_distance(fp)


def check_edit_distance(output_fp):
    '''
    Check the edit distance, compute the average edit distance
    '''
    data = read_json(output_fp)
    clean_lens = []
    noisy_lens = []
    edit_dists = []
    for id in data:
        entry = data[id]
        ref = entry['ref']
        syns = entry['syn']
        edit_avg = 0
        clean_lens.append(len(ref))
        for syn in syns:
            t = editdistance.eval(ref, syn)
            edit_avg += t
            noisy_lens.append(len(syn))
        edit_avg /= len(syns)
        edit_dists.append(edit_avg)
    plot_histogram(edit_dists)
    cl_m, cl_s, cl_max, cl_min = mean_std(clean_lens)
    nl_m, nl_s, nl_max, nl_min = mean_std(noisy_lens)
    ed_m, ed_s, ed_max, ed_min = mean_std(edit_dists)
    print('Clean strand lengths: [{:.4f}, {:.4f} ± {:.4f}, {:.4f}]'.format(cl_min, cl_m, cl_s, cl_max))
    print('Noisy strand lengths: [{:.4f}, {:.4f} ± {:.4f}, {:.4f}]'.format(nl_min, nl_m, nl_s, nl_max))
    print('Edit dist: [{:.4f}, {:.4f} ± {:.4f}, {:.4f}]'.format(ed_min, ed_m, ed_s, ed_max))


def plot_histogram(res):
    plt.hist(res, color='blue', alpha=0.7, edgecolor='black')

    # Adding title and labels
    plt.title('Histogram of averaged edit distance of each\n clean strand between its noisy version.')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()


def mean_std(data_list):
    # Convert the list to a numpy array
    data_array = np.array(data_list)

    # Calculate mean
    mean_value = np.mean(data_array)

    # Calculate standard deviation
    std_value = np.std(data_array)

    max_v = max(data_array)
    min_v = min(data_array)

    return mean_value, std_value, max_v, min_v


def check_output_length():
    output_fp = './results/test_edit_distance_new_dec/synthesized.json'
    data = read_json(output_fp)
    check_strand_length_for_dict_noise(data)


def check_strand_length_clean():
    '''
    Check the clean strand length in nanopore dataset
    '''
    dataset_dir = './data/Ours'
    data_fp = jpath(dataset_dir, 'clean.json')
    data = read_json(data_fp)
    print(data_fp)
    print(len(data))
    check_strand_length_for_dict_clean(data)


def check_strand_length():
    '''
    Check the clean strand length in nanopore dataset
    '''
    dataset_dir = './data/Microsoft_Nanopore'
    splits = ls(dataset_dir)
    for split in splits:
        if split.startswith('.'):
            continue
        print(split)
        split_fp = jpath(dataset_dir, split)
        data = read_json(split_fp)
        check_strand_length_for_dict_noise(data)


def check_strand_length_for_dict_clean(data):
    res = []
    for id in data:
        entry = data[id]
        ref = entry['ref']
        res.append(len(ref))

        # syns = entry['syn']
        # for syn in syns:
        #     res.append(len(syn))

    m, s, max_v, min_v = mean_std(res)

    print('Len: , [{:.4f}, {:.4f} ± {:.4f}, {:.4f}]'.format(min_v, m, s, max_v))


def check_strand_length_for_dict_noise(data):
    res = []
    for id in data:
        entry = data[id]
        # ref = entry['ref']
        # res.append(len(ref))

        syns = entry['syn']
        for syn in syns:
            res.append(len(syn))

    m, s, max_v, min_v = mean_std(res)

    print('Len: , [{:.4f}, {:.4f} ± {:.4f}, {:.4f}]'.format(min_v, m, s, max_v))


if __name__ == '__main__':
    _main()
