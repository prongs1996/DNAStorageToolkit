import dp_gen
import subprocess
import random
import argparse
import sys
import os
import multiprocessing as mp
import shutil
import time
from spoa import poa

bases = ['A', 'C', 'G', 'T']
blank = '-'

def majority(list):
    chars = set(list)
    chars.discard(blank)
    return max(chars, key=list.count)


def refine_majority(clu, i):
    ems = []
    for ele in clu:
        if len(ele) > i:
            ems.append(ele[i])
    if len(ems) == 0:
        ems.append(random.choice(bases))
    return ems

def NW_recover_strand(cluster, strand_len):
    c, alignment = poa(cluster, algorithm=1)

    ref_size = strand_len
    length = len(alignment[0])
    pos_to_unkown_num = {}
    for pos in range(length):
        unkown_num = 0
        for strand in alignment:
            if strand[pos] == '-':
                unkown_num = unkown_num + 1
        pos_to_unkown_num[pos] = unkown_num
    a = sorted(pos_to_unkown_num.items(), key=lambda kv:(kv[1], kv[0]), reverse = True)
    skip_index_num = length - ref_size
    skip_index = []
    loop = 0
    for pos, num in a:
        if loop == skip_index_num:
            break
        skip_index.append(pos)
        loop = loop + 1
    new_cluster = []
    for i in range(len(alignment)):
        new_cluster.append("")
    for pos in range(0, length):
        if pos in skip_index:
            continue
        for id in range(len(alignment)):
            new_cluster[id] = new_cluster[id]+alignment[id][pos]
    
    ans = ''
    recovered = ''
    for i in range(0, strand_len - 1):
        ch = majority(refine_majority(new_cluster, i))
        recovered += ch
        ans = ans[:0] + recovered
    last_ch = majority(refine_majority(new_cluster, strand_len - 1))
    ans += last_ch

    return ans

def recover_strand(cluster, strand_len):
    ans = ''
    recovered = ''

    for i in range(0, strand_len - 1):
        ch = majority(refine_majority(cluster, i))

        for j in range(len(cluster)):

            if len(cluster[j]) == i:
                cluster[j] += ch

            if cluster[j][i] != ch:

                ch2 = majority(refine_majority(cluster, i + 1))

                ch3_flag = -1
                if i + 2 < strand_len:
                    ch3_flag = 1
                    ch3 = majority(refine_majority(cluster, i + 2))

                ch4_flag = -1
                if i + 3 < strand_len:
                    ch4_flag = 1
                    ch4 = majority(refine_majority(cluster, i + 3))

                ch5_flag = -1
                if i + 4 < strand_len:
                    ch5_flag = 1
                    ch5 = majority(refine_majority(cluster, i + 4))

                if len(cluster[j]) > i + 2:
                    if cluster[j][i] == ch2 and (ch3_flag == -1 or cluster[j][i + 1] == ch3):  # erasure error
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i:]

                    elif cluster[j][i + 1] == ch and cluster[j][i + 2] == ch2:  # insertion error
                        cluster[j] = cluster[j][:i] + cluster[j][i + 1:]

                    elif cluster[j][i + 1] == ch2 and (ch3_flag == -1 or cluster[j][i + 2] == ch3):  # subs
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i + 1:]

                    elif cluster[j][i + 1] != ch2:

                        if cluster[j][i] == ch3 and (ch4_flag == -1 or cluster[j][i + 1] == ch4):  # erasure
                            cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i:]

                        elif len(cluster[j]) > i + 3:
                            if cluster[j][i + 2] == ch3 and (ch4_flag == -1 or cluster[j][i + 3] == ch4):  # subs
                                cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i + 1:]

                            elif cluster[j][i + 2] == ch and cluster[j][i + 3] == ch2:  # insertion
                                cluster[j] = cluster[j][:i] + cluster[j][i + 2:]

                            elif cluster[j][i + 1] == ch3 and (ch4_flag == -1 or cluster[j][i + 2] == ch4):
                                cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i + 1:]

                            elif cluster[j][i + 1] == ch3 and (ch4_flag == -1 or cluster[j][i + 3] == ch4):
                                cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i + 1:]

                            elif cluster[j][i + 2] == ch2 and cluster[j][i + 3] == ch3:
                                cluster[j] = cluster[j][:i] + ch + cluster[j][i + 2:]

                            elif cluster[j][i] == ch3 and (ch4_flag == -1 or cluster[j][i + 3] == ch4):
                                cluster[j] = cluster[j][:i] + ch + ch2 + ch3 + cluster[j][i + 3:]

                            elif cluster[j][i + 2] != ch3:

                                if cluster[j][i] == ch4 and (ch5_flag == -1 or cluster[j][i + 1] == ch5):  # erasure
                                    cluster[j] = cluster[j][:i] + ch + ch2 + ch3 + cluster[j][i:]

                                elif len(cluster[j]) > i + 4:
                                    if cluster[j][i + 3] == ch4 and (
                                            ch5_flag == -1 or cluster[j][i + 4] == ch5):  # subs
                                        cluster[j] = cluster[j][:i] + ch + ch2 + ch3 + cluster[j][i + 1:]

                                    elif cluster[j][i + 3] == ch and cluster[j][i + 4] == ch2:  # insertion
                                        cluster[j] = cluster[j][:i] + cluster[j][i + 3:]

                elif len(cluster[j]) == i + 2:
                    if cluster[j][i] == ch2:  # erasure error
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i:]

                    elif cluster[j][i + 1] == ch2:  # subs
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i + 1:]

                    elif cluster[j][i + 1] == ch:  # insertion error
                        cluster[j] = cluster[j][:i] + cluster[j][i + 1:]

                    else:
                        cluster[j] = cluster[j][:i] + ch

        recovered += ch
        ans = ans[:0] + recovered

    last_ch = majority(refine_majority(cluster, strand_len - 1))
    ans += last_ch

    return ans

def clean_up(file_params, strand_num, myPath):
    if myPath and os.path.exists(myPath):
       shutil.rmtree(myPath)

    if cleanup_error==1:
        if os.path.exists("results/error" + file_params + ".txt"):
            os.remove("results/error" + file_params + ".txt")
        if os.path.exists("results/error" + file_params + "_dp" + ".txt"):
            os.remove("results/error" + file_params + "_dp" + ".txt")


def reconstruct(cluster):
    strand_num=len(cluster)

    file_params = "w" + str(window_size) + "n" + str(strand_num) + "l" + str(strand_length)

    rev_cluster = []
    myPath=""

    for i in range(0, len(cluster)):
        rev_cluster.append(cluster[i][::-1])

    # Two sided BMA
    if ALG==0:
        mj = recover_strand(cluster, strand_length)
        rev_mj = recover_strand(rev_cluster, strand_length)
        rev_rev_mj = rev_mj[::-1]
        mj = mj[0:int(strand_length / 2) - 1] + rev_rev_mj[int(strand_length / 2) - 1:strand_length]


    # Single sided BMA
    elif ALG==1:
        mj = recover_strand(cluster, strand_length)
    else:
        mj = NW_recover_strand(cluster, strand_length)
    clean_up(file_params, strand_num, myPath)
    return mj

def reconstruct_clusters(clusters):
#   pool = mp.Pool(mp.cpu_count())
   pool = mp.Pool(20)
   reconstructed_strands = pool.map_async(reconstruct, clusters).get()
   pool.close()
   return reconstructed_strands

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DNA consensus')
    parser.add_argument('--i', type=str, default="ClusteredStrands.txt", help="input file")
    parser.add_argument('--o', type=str, default="ReconstructedStrands.txt", help="output file")
    parser.add_argument('--L', type=int, default=120)
    parser.add_argument('--W', type=int, default=5)
    parser.add_argument('--ALG', type=int, default=0)
    parser.add_argument('--E', type=int, default=0)
    parser.add_argument('--path', type=str)
    parser.add_argument('--coverage', type=int, default=10000)
    
    args = parser.parse_args()
    strand_length = args.L
    window_size=args.W
    ALG=args.ALG
    cleanup_error=args.E
    base_path = args.path
 	
    clusters=[]
    f=open(args.i)
    strands=f.readlines()
    f.close()	
    first=True
    clusters=[]
    cluster1=[]
    loaded=0
    coverage=args.coverage
    strands.pop(0)   
 
    for strand in strands:
        if RepresentsInt(strand.strip()):
            if not first:
                if len(cluster1)>0:
                   clusters.append(cluster1)
            else:
                first=False
            cluster1=[]
            loaded=0
        elif loaded<coverage:
            loaded=loaded+1
            cluster1.append(strand.strip())

    if len(cluster1)>1:
        clusters.append(cluster1) 

    start=time.time()
    strands=reconstruct_clusters(clusters)
    end=time.time()
    f=open(args.o, "w")
    for s in strands:
        f.write(s+"\n")
    f.close()
    print ("Trace reconstruction took " + str(end-start) +" seconds")
