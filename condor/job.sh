#!/bin/sh

CONFIG=${1}

PYTHON_ENV=/home/wzielonka/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH

echo 'START JOB (MICA training)'

module load cuda/10.1
module load gcc/4.9

echo 'ACTIVATE MICA'
source ${PYTHON_ENV}
conda activate MICA

echo 'RUN SCRIPT'
#echo 'ScriptDir' ${SCRIPT_DIR}
echo 'CONFIG: ' ${CONFIG}
cd ${SCRIPT_DIR}/..
python ./train.py --cfg ${CONFIG}