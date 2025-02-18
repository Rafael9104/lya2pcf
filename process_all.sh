#!/bin/bash
#This script executes all the necessary steps to obtain 
#the correlation, covariance and distortion matrices 
#given some delta files.

python delta_reader.py --delta-dir ./deltas
#python delta_reader.py --delta-dir ./deltas --split-number 30
#python delta_reader_eboss.py --delta-dir ./deltas

python 3pla.py --gpu --two-point
#mpirun -np 8 python 3pla.py --cpu --two-point

python post_processing.py --two-point

python distortion.py