# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import time
from glob import glob
from shutil import copyfile

logs = '/home/wzielonka/projects/MICA/testing/now/logs/'
jobs = '/home/wzielonka/projects/MICA/testing/now/jobs/'
root = '/home/wzielonka/projects/MICA/output/'

experiments = []


def test():
    global experiments
    if len(experiments) == 0:
        experiments = list(filter(lambda f: 'condor' not in f, os.listdir('../../output/')))

    os.system('rm -rf logs')
    os.system('rm -rf jobs')

    os.makedirs('logs', exist_ok=True)
    os.makedirs('jobs', exist_ok=True)

    for experiment in sorted(experiments):
        print(f'Testing {experiment}')
        copyfile(f'{root}{experiment}/model.tar', f'{root}{experiment}/best_models/best_model_last.tar')
        for idx, checkpoint in enumerate(glob(root + experiment + f'/best_models/*.tar')):
            model_name = checkpoint.split('/')[-1].split('.')[0]
            model_name = model_name.replace('best_model_', 'now_test_')
            predicted_meshes = f'{root}{experiment}/{model_name}/predicted_meshes/'
            run = f'{experiment}_{str(idx).zfill(5)}'
            with open(f'{jobs}/{run}.sub', 'w') as fid:
                fid.write('executable = /bin/bash\n')
                arguments = f'/home/wzielonka/projects/MICA/testing/now/template.sh {experiment} {checkpoint} now {predicted_meshes}'
                fid.write(f'arguments = {arguments}\n')
                fid.write(f'error = {logs}{run}.err\n')
                fid.write(f'output = {logs}{run}.out\n')
                fid.write(f'log = {logs}{run}.log\n')
                fid.write(f'request_cpus = 4\n')
                fid.write(f'request_gpus = 1\n')
                fid.write(f'requirements = (TARGET.CUDAGlobalMemoryMb > 5000) && (TARGET.CUDAGlobalMemoryMb < 33000)\n')
                fid.write(f'request_memory = 8192\n')
                fid.write(f'queue\n')

            os.system(f'condor_submit_bid 512 {jobs}/{run}.sub')

            time.sleep(2)


if __name__ == '__main__':
    test()
