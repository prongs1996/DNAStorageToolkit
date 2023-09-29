from difflib import restore
import dnacodec.trace_reconstruction.dp_gen
import subprocess
import random
import argparse
import sys
import os
import multiprocessing as mp
import shutil
import time

bases = ['A', 'C', 'G', 'T']
pbar = None

window_size = 5
strand_length = 110 # test: 120
ALG = 0
cleanup_error = 0
base_path = './test_base_path'

def majority(list):
    '''
    Return the strand that appear most times in a list of strands.
    '''
    return max(set(list), key=list.count)


def refine_majority(clu, i):
    '''
    For a list of strand,
    Take all strand that have the position i
    Take the element in their position i
    Put into a list and return

    clu: a list of strands belonging to a same cluster
    i: the position that is checking
    '''
    ems = []
    for ele in clu: # For each strand in the cluster
        if len(ele) > i: # If the length of strand has position i
            ems.append(ele[i]) # consider the element in the corresponding position
    if len(ems) == 0: 
        ems.append(random.choice(bases))
    return ems


def recover_strand_clustal(cluster, strand_len, file_params):
    ans = ''
    recovered = ''
    strand_num = len(cluster)
    i = 0
    while i < strand_len:
        ch, s_aligned = run_clustal(cluster, i, file_params)
        first_col = [s_aligned[j][0] for j in range(len(cluster))]
        if first_col == ['-'] * strand_num:
            i += 1
            continue

        for j in range(len(cluster)):
            if ch == '-' and s_aligned[j][0] != '-':  # insertion error
                cluster[j] = cluster[j][:i] + cluster[j][i + 1:]

            elif s_aligned[j][0] != ch and s_aligned[j][0] == '-':  # deletion error
                cluster[j] = cluster[j][:i] + ch + cluster[j][i:]

            elif s_aligned[j][0] != ch:  # substitution error
                cluster[j] = cluster[j][:i] + ch + cluster[j][i + 1:]

        if ch != '-':
            i += 1
            recovered += ch
            ans = ans[:0] + recovered

    while len(ans) < strand_len:
        ans += random.choice(bases)
    return ans


def recover_strand_dp(cluster, strand_len, file_params, myPath):
    ans = ''
    recovered = ''
    strand_num = len(cluster)
    i = 0
    while i < strand_len-40:
        ch, s_aligned = run_dp(cluster, i, file_params, myPath)
        first_col = [s_aligned[j][0] for j in range(len(cluster))]
        if first_col == ['-'] * strand_num:
            i += 1
            continue

        for j in range(len(cluster)):
            if ch == '-' and s_aligned[j][0] != '-':  # insertion error
                cluster[j] = cluster[j][:i] + cluster[j][i + 1:]

            elif s_aligned[j][0] != ch and s_aligned[j][0] == '-':  # deletion error
                cluster[j] = cluster[j][:i] + ch + cluster[j][i:]

            elif s_aligned[j][0] != ch:  # substitution error
                cluster[j] = cluster[j][:i] + ch + cluster[j][i + 1:]

        if ch != '-':
            i += 1
            recovered += ch
            ans = ans[:0] + recovered

    while len(ans) < strand_len:
        ans += random.choice(bases)
    return ans


def run_clustal(cluster, i, file_params):
    c_id = str(random.randint(0, 100000000))
    filename = "test_step_" + str(i) + file_params + c_id+".txt"
    resname = "res_step_" + str(i) + file_params + c_id+".txt"
    f = open(filename, "w+")
    j = 0
    empty = []
    for strand in cluster:
        xx = strand[i:min(i + window_size, strand_length)]
        if xx.strip():
            f.write(">C"+str(j)+"\n" +
                    strand[i:min(i + window_size, strand_length)] + "\n")
        else:
            empty.append(j)
        j = j+1
    f.close()
    strand_num = len(cluster)

    if len(empty)+1 < len(cluster):
        subprocess.run(["./clustalw2", "-INFILE="+filename, "-OUTPUT=fasta", "-OUTFILE="+resname, "-PWDNAMATRIX=IUB",
                       "-DNAMATRIX=IUB", "-TYPE=DNA", "-PWGAPOPEN=1.9", "-PWGAPEXT=1.9", "-GAPOPEN=1.9", "-GAPEXT=1.9"], check=True)
        #subprocess.run(["./clustalw2","-INFILE="+filename,"-OUTPUT=fasta", "-OUTFILE="+resname,  "-ITERATION=TREE", "-NUMITER=3"], check=True)
    else:
        subprocess.run(["cp", filename, resname], check=True)

    s_aligned1 = []
    if os.path.exists(resname):
        with open(resname) as g:
            s_aligned1 = g.readlines()
        g.close()
        s_aligned1 = [x.strip() for x in s_aligned1]
        s_aligned = []
        for x in s_aligned1:
            if not x.startswith('>'):
                s_aligned.append(x)
        for j in empty:
            s_aligned.insert(j, '-')

    subprocess.run(["cat", filename], check=True)
    print("")
    subprocess.run(["cat", resname], check=True)
    #input("Press Enter to continue...")
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(resname):
        os.remove(resname)
    if os.path.exists("test_step_" + str(i) + file_params + c_id+".dnd"):
        os.remove("test_step_" + str(i) + file_params + c_id+".dnd")

    while s_aligned != [''] * strand_num:
        first_col = [s_aligned[j][0] for j in range(len(cluster))]
        if first_col == ['-'] * strand_num:
            for j in range(len(cluster)):
                s_aligned[j] = s_aligned[j][1:]
        else:
            break

    ch = majority(refine_majority(s_aligned, 0))
    if s_aligned == [''] * strand_num:
        s_aligned = [ch] * strand_num
    return ch, s_aligned


def run_dp(cluster, i, file_params, myPath):
    filename = myPath + "/test" + file_params + ".txt"
    f = open(filename, "w+")
    for strand in cluster:
        f.write(strand[i:min(i + window_size, strand_length)] + "\n")
    f.close()
    strand_num = len(cluster)

    resname = myPath + "/res" + file_params + ".txt"
    subprocess.run([myPath + "/dp" + str(strand_num),
                   file_params, myPath], check=True)

    s_aligned = []
    if os.path.exists(resname):
        with open(resname) as g:
            s_aligned = g.readlines()
        s_aligned = [x.strip() for x in s_aligned]
        g.close()

    while s_aligned != [''] * strand_num:
        first_col = [s_aligned[j][0] for j in range(len(cluster))]
        if first_col == ['-'] * strand_num:
            for j in range(len(cluster)):
                s_aligned[j] = s_aligned[j][1:]
        else:
            break

    ch = majority(refine_majority(s_aligned, 0))
    if s_aligned == [''] * strand_num:
        s_aligned = [ch] * strand_num
    return ch, s_aligned


def recover_strand(cluster, strand_len):
    '''
    cluster: a list of strands that belong to a same cluster
    strand_len: the desired strand length, = 120
    '''
    ans = '' # the cluster consensus
    recovered = ''

    for i in range(0, strand_len - 1): # For every bit in the consensus to be genrated
        ch = majority(refine_majority(cluster, i)) # For all strands that have position i, choose the element that appear most time, denote as ch

        for j in range(len(cluster)): # For each strand in the cluster

            if len(cluster[j]) == i: # If the strand doesn't have position i, make it a little bit longer by append 'ch', then the i-th element will be 'ch'
                cluster[j] += ch

            if cluster[j][i] != ch: # If the strand have position i so is not appended 'ch' in the end

                ch2 = majority(refine_majority(cluster, i + 1)) # For all strands that have position i+1, choose the element that appear most time at (i+1), denote as ch2

                ch3_flag = -1
                if i + 2 < strand_len: # If we still can consider position i+2 (append 2 characters to the position we are consider, the total length still won't exceed the maximum constraint)
                    ch3_flag = 1 
                    ch3 = majority(refine_majority(cluster, i + 2))

                ch4_flag = -1
                if i + 3 < strand_len: # If we still can consider position i+3
                    ch4_flag = 1
                    ch4 = majority(refine_majority(cluster, i + 3))

                ch5_flag = -1
                if i + 4 < strand_len: # If we still can consider position i+4
                    ch5_flag = 1
                    ch5 = majority(refine_majority(cluster, i + 4))

                if len(cluster[j]) > i + 2: # 
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
        ans = ans[:0] + recovered # copy the value of 'recovered' to 'ans'

    last_ch = majority(refine_majority(cluster, strand_len - 1))
    ans += last_ch
    # print(len(ans), ans)

    return ans


def compile_dp(strand_num, myPath):
    subprocess.check_call('mkdir -p '+myPath, shell=True)
    file_name = myPath + "/dp" + str(strand_num) + ".cpp"
    f = open(file_name, "w+")
    f.write(dp_gen.generate_msa(strand_num))
    f.close()
    cmd = 'g++ -o3 ' + myPath + '/dp' + \
        str(strand_num) + '.cpp -o ' + myPath + '/dp' + str(strand_num)
    subprocess.check_call(cmd, shell=True)


def clean_up(file_params, strand_num, myPath):
    if myPath and os.path.exists(myPath):
        shutil.rmtree(myPath)

    if cleanup_error == 1:
        if os.path.exists("results/error" + file_params + ".txt"):
            os.remove("results/error" + file_params + ".txt")
        if os.path.exists("results/error" + file_params + "_dp" + ".txt"):
            os.remove("results/error" + file_params + "_dp" + ".txt")


def reconstruct(cluster):
    id, cluster = cluster
    # print(i, cluster)
    # global pbar
    # pbar.update(1)

    strand_num = len(cluster)

    file_params = "w" + str(window_size) + "n" + \
        str(strand_num) + "l" + str(strand_length)

    rev_cluster = []
    myPath = ""

    for i in range(0, len(cluster)):
        rev_cluster.append(cluster[i][::-1])

    if ALG == 0:
        mj = recover_strand(cluster, strand_length)
        rev_mj = recover_strand(rev_cluster, strand_length)

    elif ALG == 1:
        c_id = str(random.randint(0, 100000000))
        myPath = base_path + "/consensus_data/c_" + c_id
        compile_dp(strand_num, myPath)
        mj = recover_strand_dp(cluster, strand_length, file_params, myPath)
        rev_mj = recover_strand_dp(
            rev_cluster, strand_length, file_params, myPath)

    else:
        mj = recover_strand_clustal(cluster, strand_length, file_params)
        rev_mj = recover_strand_clustal(
            rev_cluster, strand_length, file_params)

    rev_rev_mj = rev_mj[::-1]
    mj = mj[0:int(strand_length / 2) - 1] + \
        rev_rev_mj[int(strand_length / 2) - 1:strand_length]

    clean_up(file_params, strand_num, myPath)
    return id, mj


TOTAL_SUCCESSES = []


def callback(successes):
    global TOTAL_SUCCESSES

    for entry in successes:
        TOTAL_SUCCESSES.append(entry)


class PoolProgress:
    def __init__(self, pool, update_interval=3):
        self.pool = pool
        self.update_interval = update_interval

    def track(self, job):
        task = self.pool._cache[job._job]
        while task._number_left > 0:
            print("Tasks remaining = {0}".format(
                task._number_left*task._chunksize))
            time.sleep(self.update_interval)


def reconstruct_clusters(clusters):
    # print('len cluster:', len(clusters))
    # print([len(i) for i in clusters])

    t11 = time.time()

    from tqdm import tqdm
    from time import sleep

    # global pbar
    # pbar = tqdm(total=len(clusters))
    results = []
    with mp.Pool(200) as pool:
        pp  = PoolProgress(pool)
        res = pool.map_async(reconstruct, clusters)
        pp.track(res)
        reconstructed_strands = res.get()
    
    t12 = time.time()

    return reconstructed_strands


def reconstruct_clusters2(clusters):
    strands = []
    i = 0
    for c in clusters:
        strands.append(reconstruct(c, i))
        i = i+1
    return strands


def reconstruct_clusters3(clusters):
    reconstructed_strands = []
    for x in clusters:
        reconstructed_strands.append(reconstruct(x))
    return reconstructed_strands


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DNA consensus')
    parser.add_argument(
        '--i', type=str, default="ClusteredStrands.txt", help="input file")
    parser.add_argument(
        '--o', type=str, default="ReconstructedStrands.txt", help="output file")
    parser.add_argument('--L', type=int, default=120)
    parser.add_argument('--W', type=int, default=5)
    parser.add_argument('--ALG', type=int, default=0)
    parser.add_argument('--E', type=int, default=0)
    parser.add_argument('--path', type=str)
    parser.add_argument('--coverage', type=int, default=10000)

    args = parser.parse_args()
    strand_length = args.L
    window_size = args.W
    ALG = args.ALG
    cleanup_error = args.E
    base_path = args.path

    clusters = []
    f = open(args.i)
    strands = f.readlines()
    f.close()
    first = True
    clusters = []
    cluster1 = []
    loaded = 0
    coverage = args.coverage
    strands.pop(0)


    for strand in strands:
        if RepresentsInt(strand.strip()): # we don't care about int. Int value are either total number of clusters, or represent number of strands in a batch
            if not first: # count numbe
                if len(cluster1) > 0:
                    clusters.append(cluster1) # add this cluster to a list of clusters
            else:
                first = False
            cluster1 = []
            loaded = 0
        elif loaded < coverage:
            loaded = loaded+1
            cluster1.append(strand.strip()) # add strand to a cluster

    # Debug: count the largest cluster and average cluster size
    tot = 0
    m = 0
    minimal = 100
    for cluster in clusters:
        if len(cluster) > m:
            m = len(cluster)
        if len(cluster) < minimal:
            minimal = len(cluster) 
        tot += len(cluster)
    avg = tot / len(clusters)
    print("There are {} clusters in total.".format(len(clusters)))
    print('Avg cluster size: {}, Max cluster size: {}, Min cluster size: {}'.format(avg, m, minimal))

    # Remove clusters that are too large
    clusters_old = clusters
    clusters = []
    for cluster in clusters_old:
        if len(cluster) <= 20:
            clusters.append(cluster)

    tot = 0
    m = 0
    for cluster in clusters:
        if len(cluster) > m:
            m = len(cluster)
        tot += len(cluster)
    avg = tot / len(clusters)
    print('After removing,')
    print("There are {} clusters in total.".format(len(clusters)))
    print('Avg cluster size: {}, Max cluster size: {}'.format(avg, m))

    t2 = time.time()

    if len(cluster1) > 1:
        clusters.append(cluster1)

    t3 = time.time()

    start = time.time()
    strands = reconstruct_clusters(clusters)
    end = time.time()

    t4 = time.time()

    f = open(args.o, "w")
    for s in strands:
        f.write(s+"\n")
    f.close()
    print("Trace reconstruction took " + str(end-start) + " seconds")

    t5 = time.time()
