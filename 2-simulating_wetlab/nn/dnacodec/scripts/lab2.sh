#!/bin/sh
set -x

BASE_DIR='/scratch/cs6219/e0740922/DNA_data'  # copy this folder to your home folder and update this line correspondingly

CODE_DIR='/scratch/cs6219/e0740922/dnacodec'  # copy this folder to your home folder and update this line correspondingly

#export PATH="/scratch/shared/cs6219/dependencies/GCC/bin:/scratch/shared/cs6219/dependencies/squashfs-root/usr/bin:$PATH"
#export LD_LIBRARY_PATH=/scratch/shared/cs6219/dependencies/GCC/lib64/:$LD_LIBRARY_PATH

#input configuration file
config_file='mystery'

skipRS='0'

#probability of error of each type (insertions, deletions, substitutions)
P='0.01'

coverage='10'

#encode
cd $CODE_DIR/codec
python3 $CODE_DIR/codec/codec.py $BASE_DIR $CODE_DIR/codec/configs/${config_file}.cfg 0 $config_file $skipRS 
cd -

#create $coverage number of noisy copies for each strand and shuffle them
mkdir -p $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}
python3 $CODE_DIR/scripts/noise.py --N ${coverage}  --subs $P --dels $P --inss $P  --i $BASE_DIR/outputs/${config_file}/EncodedStrands.txt  --o $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/UnderlyingClusters.txt
python3 $CODE_DIR/scripts/shuffle.py $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/UnderlyingClusters.txt  $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/NoisyStrands.txt


cd $CODE_DIR/dna_clustering
cp $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/NoisyStrands.txt input/.
cp $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/UnderlyingClusters.txt input/.
make run

cp $CODE_DIR/dna_clustering/output/ClusteredStrands.txt  $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/.
 

python3 $CODE_DIR/trace_reconstruction/recon.py --i $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/ClusteredStrands.txt  --o  $BASE_DIR/outputs/${config_file}_P${P}_N${coverage}/ReconstructedStrands.txt --path /dev/shm 
 
cd $CODE_DIR/codec
python3 $CODE_DIR/codec/codec.py $BASE_DIR $CODE_DIR/codec/configs/${config_file}.cfg 1 ${config_file}_P${P}_N${coverage} $skipRS 
cd -
