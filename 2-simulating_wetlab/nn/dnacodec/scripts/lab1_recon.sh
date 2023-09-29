#!/bin/sh
set -x

BASE_DIR='/scratch/cs6219/e0740922/DNA_data'  # copy this folder to your home folder and update this line correspondingly

CODE_DIR='/scratch/cs6219/e0740922/dnacodec'  # copy this folder to your home folder and update this line correspondingly

#export PATH="/scratch/shared/cs6219/dependencies/GCC/bin:/scratch/shared/cs6219/dependencies/squashfs-root/usr/bin:$PATH"
#export LD_LIBRARY_PATH=/scratch/shared/cs6219/dependencies/GCC/lib64/:$LD_LIBRARY_PATH

#input configuration file
config_file='cat_naive'

skipRS='0' # use ECC

# Probability of error of each type (insertions, deletions, substitutions)
P='0.01'

coverage='10'

# # 1. Encode
# start=`date +%s`
# cd $CODE_DIR/codec
# python3 $CODE_DIR/codec/codec.py $BASE_DIR $CODE_DIR/codec/configs/${config_file}.cfg 0 $config_file $skipRS 
# cd -
# end=`date +%s`
# time=`echo $start $end | awk '{print $2-$1}'`
# echo $time seconds for encoding

# # 2. Create $coverage number of noisy copies for each strand and shuffle them
# start=`date +%s`
# mkdir -p $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}
# python3 $CODE_DIR/scripts/noise.py --N ${coverage}  --subs $P --dels $P --inss $P  --i $BASE_DIR/outputs/${config_file}/EncodedStrands.txt  --o $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/UnderlyingClusters.txt
# python3 $CODE_DIR/scripts/shuffle.py $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/UnderlyingClusters.txt  $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/NoisyStrands.txt
# end=`date +%s`
# time=`echo $start $end | awk '{print $2-$1}'`
# echo $time seconds for noise

# cd $CODE_DIR/dna_clustering
# cp $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/NoisyStrands.txt input/.
# cp $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/UnderlyingClusters.txt input/.
# make run

# cp $CODE_DIR/dna_clustering/output/ClusteredStrands.txt  $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/.
 
# 3. Trace reconstruction
start=`date +%s`
python3 $CODE_DIR/trace_reconstruction/recon.py \
        --i $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/ClusteredStrands.txt  \
        --o  $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/ReconstructedStrands.txt \
        --path /dev/shm \
        --ALG 0
end=`date +%s`
time=`echo $start $end | awk '{print $2-$1}'`
echo $time seconds for trace reconstruction

# # 4. Decode
# start=`date +%s`
# cd $CODE_DIR/codec
# python3 $CODE_DIR/codec/codec.py $BASE_DIR $CODE_DIR/codec/configs/${config_file}.cfg 1 ${config_file}_P${P}_N${coverage} $skipRS 
# cd -
# end=`date +%s`
# time=`echo $start $end | awk '{print $2-$1}'`
# echo $time seconds for decoding
