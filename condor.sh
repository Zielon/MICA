#!/bin/sh

# bash condor.sh 100 ./configs/mica.yml 1

# default parameters
BID=3
CONFIG=./configs/mica.yml
NODE_CONFIG=condor/config.sub
NODE_SCRIPT=./condor/job.sh
GPUS=1
GPU_TYPE=0

# set parameters
if [ -n "$1" ]; then BID=${1}; fi
if [ -n "$2" ]; then CONFIG=${2}; fi
if [ -n "$3" ]; then GPU_TYPE=${3}; fi
if [ -n "$4" ]; then GPUS=${4}; fi
if [ -n "$5" ]; then NODE_CONFIG=${5}; fi
if [ -n "$6" ]; then NODE_SCRIPT=${6}; fi

mkdir -p output/condor_logs
cp -nf ${NODE_CONFIG}{,.bak}

GPU_NAME=Error

if [ $GPU_TYPE -eq 0 ]; then GPU_NAME='Quadro RTX 6000'; fi
if [ $GPU_TYPE -eq 1 ]; then GPU_NAME='Tesla V100-SXM2-32GB'; fi
if [ $GPU_TYPE -eq 2 ]; then GPU_NAME='NVIDIA GeForce RTX 2080 Ti'; fi

NAME=$(basename ${CONFIG} .yml)
sed -i "s/{errorfile}/${NAME}/" ${NODE_CONFIG}.bak
sed -i "s/{outfile}/${NAME}/" ${NODE_CONFIG}.bak
sed -i "s/{logfile}/${NAME}/" ${NODE_CONFIG}.bak
sed -i "s/{gpus}/${GPUS}/" ${NODE_CONFIG}.bak
sed -i "s/{gpu_name}/${GPU_NAME}/" ${NODE_CONFIG}.bak

# start node and execute script
echo 'Executing:' ${NODE_SCRIPT} ${CONFIG}
echo '# BID:' ${BID}
echo '# GPUS:' ${GPUS}
echo '# GPU NAME:' ${GPU_NAME}

condor_submit_bid ${BID} ${NODE_CONFIG}.bak -append "arguments = ${NODE_SCRIPT} ${CONFIG}"
rm ${NODE_CONFIG}.bak
