#!/bin/bash
# This script is used to run every shuffled data set

save_dir='/home/josue/cosmologia/three_point/3pla/outputs/dr14.1/'

# exit 0  
#echo 'Running ordered data set'
#python 2pla.py --from-extracted --cuda --two-point --save-dir=$save_dir

for i in $(seq 21 1 2000)
do
echo "Running $i shuffled data set"
python shuffler.py --number=${i}
# mkdir ${save_dir}"/shuffle_ra_dec_"${i}"/"
python 2pla.py --from-extracted --cuda --two-point --save-dir=${save_dir}"/shuffle_ra_dec_"${i}"/"
rm ${save_dir}"/shuffle_ra_dec_"${i}"/data.npy"
done
