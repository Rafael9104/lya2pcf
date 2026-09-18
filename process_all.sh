#!/bin/bash
#This script executes all the necessary steps to obtain
#the correlation, covariance and distortion matrices
#given some delta files.
#
#Requires the package to be installed first: pip install -e . (or
#pip install -e .[gpu] for the GPU path), which provides the
#lya2pcf-* commands used below.

lya2pcf-extract --delta-dir ./deltas
#lya2pcf-extract --delta-dir ./deltas --split-number 30
#lya2pcf-extract-eboss --delta-dir ./deltas

lya2pcf-correlate --gpu
#mpirun -np 8 lya2pcf-correlate --cpu

lya2pcf-post

lya2pcf-distort
