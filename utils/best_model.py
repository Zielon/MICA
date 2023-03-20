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

import numpy as np
from loguru import logger


class BestModel:
    def __init__(self, trainer):
        self.average = np.Inf
        self.weighted_average = np.Inf
        self.smoothed_average = np.Inf
        self.smoothed_weighted_average = np.Inf
        self.running_average = np.Inf
        self.running_weighted_average = np.Inf
        self.now_mean = None

        self.trainer = trainer
        self.counter = None

        self.N = trainer.cfg.running_average

        os.makedirs(os.path.join(self.trainer.cfg.output_dir, 'best_models'), exist_ok=True)

    def state_dict(self):
        return {
            'average': self.average,
            'smoothed_average': self.smoothed_average,
            'running_average': self.running_average,
            'now_mean': self.now_mean,
            'counter': self.counter,
        }

    def load_state_dict(self, dict):
        self.average = dict['average']
        self.smoothed_average = dict['smoothed_average']
        self.running_average = dict['running_average']
        self.now_mean = dict['now_mean']
        self.counter = dict['counter']

        logger.info(f'[BEST] Best score weighted average: '
                    f'NoW mean: {self.now_mean:.6f} | '
                    f'average: {self.average:.6f} | '
                    f'smoothed average: {self.running_average:.6f}')

    def __call__(self, weighted_average, average):
        if self.counter is None:
            self.counter = 1
            self.average = average
            self.weighted_average = weighted_average
            self.running_weighted_average = weighted_average
            self.running_average = average

            return weighted_average, average

        if weighted_average < self.weighted_average:
            delta = self.weighted_average - weighted_average
            self.weighted_average = weighted_average
            logger.info(f'[BEST] (Average weighted)   {self.trainer.global_step} | {delta:.6f} improvement and value: {self.weighted_average:.6f}')
            self.trainer.save_checkpoint(os.path.join(self.trainer.cfg.output_dir, 'best_models', f'best_model_0.tar'))

        if average < self.average:
            delta = self.average - average
            self.average = average
            logger.info(f'[BEST] (Average)   {self.trainer.global_step} | {delta:.6f} improvement and value: {self.average:.6f}')
            self.trainer.save_checkpoint(os.path.join(self.trainer.cfg.output_dir, 'best_models', f'best_model_1.tar'))

        n = self.N

        self.running_average = self.running_average * ((n - 1) / n) + (average / n)
        if self.running_average < self.smoothed_average:
            delta = self.smoothed_average - self.running_average
            self.smoothed_average = self.running_average
            logger.info(f'[BEST] (Average Smoothed) {self.trainer.global_step} | {delta:.6f} improvement and value: {self.smoothed_average:.6f} | counter: {self.counter} | window: {n}')
            self.trainer.save_checkpoint(os.path.join(self.trainer.cfg.output_dir, 'best_models', f'best_model_3.tar'))

        self.counter += 1

        return self.running_weighted_average, self.running_average

    def now(self, median, mean, std):
        if self.now_mean is None:
            self.now_mean = mean
            return

        if mean < self.now_mean:
            delta = self.now_mean - mean
            self.now_mean = mean
            logger.info(f'[BEST] (NoW)   {self.trainer.global_step} | {delta:.6f} improvement and mean: {self.now_mean:.6f} std: {std} median: {median}')
            self.trainer.save_checkpoint(os.path.join(self.trainer.cfg.output_dir, 'best_models', f'best_model_now.tar'))
