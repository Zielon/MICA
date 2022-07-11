#!/bin/bash

PYTHON_ENV=/home/wzielonka/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH
export LD_LIBRARY_PATH=/is/software/nvidia/nccl-2.4.8-cuda10.1/lib/

source ${PYTHON_ENV}
module load cuda/10.1
module load gcc/4.9

EXPERIMENT=''
CHECKPOINT=''
BENCHMARK=''
PREDICTED=''

echo 'Testing has started...'

if [ -n "$1" ]; then EXPERIMENT=${1}; fi
if [ -n "$2" ]; then CHECKPOINT=${2}; fi
if [ -n "$3" ]; then BENCHMARK=${3}; fi
if [ -n "$4" ]; then PREDICTED=${4}; fi

ROOT=/home/wzielonka/projects/MICA/output/
NOW=/home/wzielonka/datasets/NoWDataset/final_release_version/

conda activate NFC

cd /home/wzielonka/projects/MICA
python test.py --cfg /home/wzielonka/projects/MICA/configs/${EXPERIMENT}.yml --test_dataset ${BENCHMARK} --checkpoint ${CHECKPOINT}

source /home/wzielonka/.virtualenvs/NoW/bin/activate
cd /home/wzielonka/projects/NoW
python compute_error.py ${NOW} ${PREDICTED} true

# Plot diagram
# source /home/wzielonka/.virtualenvs/NoW/bin/activate
# python cumulative_errors.py