# Let's assume this one goes normal reed-solomon, RNN wetlab, clustering with w-gram, reconstruction and decoding

#!/bin/sh
set -x

# Set up directories

BASE_DIR='/home/csbingao/FAST/DNAPipeline/data'  # copy this folder to your home folder and update this line correspondingly

CODE_DIR='/home/csbingao/FAST/DNAPipeline'  # copy this folder to your home folder and update this line correspondingly

#export PATH="/home/old_home/shared/cs6219/dependencies/GCC/bin:/scratch/shared/cs6219/dependencies/squashfs-root/usr/bin:$PATH"
#export LD_LIBRARY_PATH=/home/old_home/shared/cs6219/dependencies/GCC/lib64/:$LD_LIBRARY_PATH


# The DNA based data storage pipeline consists of the following steps:
# 1) Encoding data into DNA strands
# 2) Synthesizing, storing and sequencing DNA strands - wetlab activities which introduce errors
# 3) Clustering sequenced reads
# 4) Reconstructing original DNA strands from clusters of reads 
# 5) Decoding data from reconstructed strands
# 
# We will be implementing the entire pipeline here:

# Pick configuration file for Input Data

file_name='cat_naive'
config_file='1-encoding-decoding_config'
# for example-1 use "1-encoding-decoding_config.cfg", for example-2 use "text-data-example.cfg"

skipRS='0' # '0' if you want to use Reed Solomon Code for redundancy, '1' if not

# Probability of error of each type (insertions, deletions, substitutions)
P='0.01'

# Number of reads per DNA strand
coverage='10'

# 1) Encoding data into DNA strands


cd $CODE_DIR/1-encoding-decoding/
python3 $CODE_DIR/1-encoding-decoding/codec.py $BASE_DIR $CODE_DIR/0-configs/${config_file}.cfg 0 $file_name $skipRS 
cd -


# 2) Simulating wetlab activities which introduce errors - Synthesizing, storing and sequencing DNA strands - 


mkdir -p $BASE_DIR/output/${file_name}_P${P}_N${coverage}
python3 $CODE_DIR/2-simulating_wetlab/naive/noise.py --N ${coverage}  --subs $P --dels $P --inss $P  --i $BASE_DIR/output/${file_name}/EncodedStrands.txt  --o $BASE_DIR/output/${file_name}_P${P}_N${coverage}/UnderlyingClusters.txt
python3 $CODE_DIR/2-simulating_wetlab/naive/shuffle.py $BASE_DIR/output/${file_name}_P${P}_N${coverage}/UnderlyingClusters.txt  $BASE_DIR/output/${file_name}_P${P}_N${coverage}/NoisyStrands.txt


# 3) Clustering sequenced reads
 

cd $CODE_DIR/3-clustering
cp $BASE_DIR/output/${file_name}_P${P}_N${coverage}/NoisyStrands.txt input/.
cp $BASE_DIR/output/${file_name}_P${P}_N${coverage}/UnderlyingClusters.txt input/.
make run

cp $CODE_DIR/3-clustering/output/ClusteredStrands.txt  $BASE_DIR/output/${file_name}_P${P}_N${coverage}/.


# 4) Reconstructing original DNA strands from clusters of reads 


python3 $CODE_DIR/4-reconstruction/recon.py --i $BASE_DIR/output/${file_name}_P${P}_N${coverage}/ClusteredStrands.txt  --o  $BASE_DIR/output/${file_name}_P${P}_N${coverage}/ReconstructedStrands.txt --coverage 10 --path /dev/shm 

# 5) Decoding data from reconstructed strands



cd $CODE_DIR/1-encoding-decoding
python3 $CODE_DIR/1-encoding-decoding/codec.py $BASE_DIR $CODE_DIR/0-configs/${config_file}.cfg 1 ${file_name}_P${P}_N${coverage} $skipRS 
cd -
