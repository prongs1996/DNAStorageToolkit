import random
import sys

in_file = open(sys.argv[1], "r")
l1 = in_file. readlines()
in_file.close()
random.shuffle(l1)
out_file = open(sys.argv[2], "w")
for l in l1:
  if l[0:7] != "CLUSTER":
     out_file.write(l.strip()+"\n")
out_file.close()

