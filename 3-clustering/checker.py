import random
import editdistance
import itertools
import csv
import sys
import time
def accuracy1(output_clusters,underlying_clusters):
   M={}
   for i in range(len(output_clusters)):
      for dna_strand in output_clusters[i]:
         if dna_strand not in M:
            M[dna_strand] = []
         M[dna_strand].append(i)
   count =0
   for underlying_cluster in underlying_clusters:
      freq={}
      checked={}
      for dna_strand in underlying_cluster:
         if dna_strand in checked:
            continue
         checked[dna_strand] = 0
         for indexes in M[dna_strand]:
            if indexes not in freq:
               freq[indexes] = 0
            freq[indexes] += 1
         for cluster_index,frequency in freq.items():
            if len(output_clusters[cluster_index])==frequency and len(output_clusters[cluster_index])/len(underlying_cluster) >= 1:
               count += 1
               break
   return count/len(underlying_clusters)

def main():
    clusters = []
    out = "output/" + sys.argv[1]+ "/output.txt"
    with open(out, "r") as f:
         no = int(f.readline())
         for i in range(no):
            tme = int(f.readline())
            cluster = []
            for j in range(tme):
                cluster.append(f.readline().strip('\n'))
            clusters.append(cluster)

    underlying_clusters = []
    tmp = "input/" + sys.argv[1] + "/underlying.txt"
    with open(tmp , "r") as f:
        f.readline()
        for i in range(0 , 65535):
            cluster = []
            while True:
                s = f.readline().strip('\n')
                if not s:
                   break
                if s[0:7] == "CLUSTER": 
                   continue
                cluster.append(s)
            underlying_clusters.append(cluster)
    print(accuracy1(clusters , underlying_clusters))
if __name__ == "__main__":
    main()

