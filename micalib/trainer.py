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
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from configs.config import cfg
from utils import util

sys.path.append("./micalib")
from validator import Validator


def print_info(rank):
    props = torch.cuda.get_device_properties(rank)

    logger.info(f'[INFO]            {torch.cuda.get_device_name(rank)}')
    logger.info(f'[INFO] Rank:      {str(rank)}')
    logger.info(f'[INFO] Memory:    {round(props.total_memory / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Allocated: {round(torch.cuda.memory_allocated(rank) / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Cached:    {round(torch.cuda.memory_reserved(rank) / 1024 ** 3, 1)} GB')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer(object):
    def __init__(self, nfc_model, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.K = self.cfg.dataset.K

        # deca model
        self.nfc = nfc_model.to(self.device)

        self.validator = Validator(self)
        self.configure_optimizers()
        self.load_checkpoint()

        # reset optimizer if loaded from pretrained model
        if self.cfg.train.reset_optimizer:
            self.configure_optimizers()  # reset optimizer
            logger.info(f"[TRAINER] Optimizer was reset")

        if self.cfg.train.write_summary and self.device == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

        print_info(device)

    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            params=self.nfc.parameters_to_optimize(),
            amsgrad=False)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.1)

    def load_checkpoint(self):
        self.epoch = 0
        self.global_step = 0
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        model_path = os.path.join(self.cfg.output_dir, 'model.tar')
        if os.path.exists(self.cfg.pretrained_model_path):
            model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'opt' in checkpoint:
                self.opt.load_state_dict(checkpoint['opt'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f"[TRAINER] Resume training from {model_path}")
            logger.info(f"[TRAINER] Start from step {self.global_step}")
            logger.info(f"[TRAINER] Start from epoch {self.epoch}")
        else:
            logger.info('[TRAINER] Model path not found, start training from scratch')

    def save_checkpoint(self, filename):
        if self.device == 0:
            model_dict = self.nfc.model_dict()

            model_dict['opt'] = self.opt.state_dict()
            model_dict['scheduler'] = self.scheduler.state_dict()
            model_dict['validator'] = self.validator.state_dict()
            model_dict['epoch'] = self.epoch
            model_dict['global_step'] = self.global_step
            model_dict['batch_size'] = self.batch_size

            torch.save(model_dict, filename)

    def training_step(self, batch):
        self.nfc.train()

        images = batch['image'].to(self.device)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        flame = batch['flame']
        arcface = batch['arcface']
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device)

        inputs = {
            'images': images,
            'dataset': batch['dataset'][0]
        }

        encoder_output = self.nfc.encode(images, arcface)
        encoder_output['flame'] = flame

        decoder_output = self.nfc.decode(encoder_output, self.epoch)
        losses = self.nfc.compute_losses(inputs, encoder_output, decoder_output)

        all_loss = 0.
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]

        losses['all_loss'] = all_loss

        opdict = \
            {
                'images': images,
                'flame_verts_shape': decoder_output['flame_verts_shape'],
                'pred_canonical_shape_vertices': decoder_output['pred_canonical_shape_vertices'],
            }

        if 'deca' in decoder_output:
            opdict['deca'] = decoder_output['deca']

        return losses, opdict

    def validation_step(self):
        self.validator.run()

    def evaluation_step(self):
        pass

    def prepare_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.device)

        self.train_dataset, total_images = datasets.build_train(self.cfg.dataset, self.device)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator)

        self.train_iter = iter(self.train_dataloader)
        logger.info(f'[TRAINER] Training dataset is ready with {len(self.train_dataset)} actors and {total_images} images.')

    def fit(self):
        self.prepare_data()
        iters_every_epoch = int(len(self.train_dataset) / self.batch_size)
        max_epochs = int(self.cfg.train.max_steps / iters_every_epoch)
        start_epoch = self.epoch
        for epoch in range(start_epoch, max_epochs):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch + 1}/{max_epochs}]"):
                if self.global_step > self.cfg.train.max_steps:
                    break
                try:
                    batch = next(self.train_iter)
                except Exception as e:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                visualizeTraining = self.global_step % self.cfg.train.vis_steps == 0

                self.opt.zero_grad()
                losses, opdict = self.training_step(batch)

                all_loss = losses['all_loss']
                all_loss.backward()
                self.opt.step()
                self.global_step += 1

                if self.global_step % self.cfg.train.log_steps == 0 and self.device == 0:
                    loss_info = f"\n" \
                                f"  Epoch: {epoch}\n" \
                                f"  Step: {self.global_step}\n" \
                                f"  Iter: {step}/{iters_every_epoch}\n" \
                                f"  LR: {self.opt.param_groups[0]['lr']}\n" \
                                f"  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'  {k}: {v:.4f}\n'
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/' + k, v, global_step=self.global_step)
                    logger.info(loss_info)

                if visualizeTraining and self.device == 0:
                    visdict = {
                        'input_images': opdict['images'],
                    }
                    # add images to tensorboard
                    for k, v in visdict.items():
                        self.writer.add_images(k, np.clip(v.detach().cpu(), 0.0, 1.0), self.global_step)

                    pred_canonical_shape_vertices = torch.empty(0, 3, 512, 512).cuda()
                    flame_verts_shape = torch.empty(0, 3, 512, 512).cuda()
                    deca_images = torch.empty(0, 3, 512, 512).cuda()
                    input_images = torch.empty(0, 3, 224, 224).cuda()
                    L = opdict['pred_canonical_shape_vertices'].shape[0]
                    S = 4 if L > 4 else L
                    for n in np.random.choice(range(L), size=S, replace=False):
                        rendering = self.nfc.render.render_mesh(opdict['pred_canonical_shape_vertices'][n:n + 1, ...])
                        pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
                        rendering = self.nfc.render.render_mesh(opdict['flame_verts_shape'][n:n + 1, ...])
                        flame_verts_shape = torch.cat([flame_verts_shape, rendering])
                        input_images = torch.cat([input_images, opdict['images'][n:n + 1, ...]])
                        if 'deca' in opdict:
                            deca = self.nfc.render.render_mesh(opdict['deca'][n:n + 1, ...])
                            deca_images = torch.cat([deca_images, deca])

                    visdict = {}

                    if 'deca' in opdict:
                        visdict['deca'] = deca_images

                    visdict["pred_canonical_shape_vertices"] = pred_canonical_shape_vertices
                    visdict["flame_verts_shape"] = flame_verts_shape
                    visdict["images"] = input_images

                    savepath = os.path.join(self.cfg.output_dir, 'train_images/train_' + str(epoch) + '.jpg')
                    util.visualize_grid(visdict, savepath, size=512)

                if self.global_step % self.cfg.train.val_steps == 0:
                    self.validation_step()

                if self.global_step % self.cfg.train.lr_update_step == 0:
                    self.scheduler.step()

                if self.global_step % self.cfg.train.eval_steps == 0:
                    self.evaluation_step()

                if self.global_step % self.cfg.train.checkpoint_steps == 0:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))

                if self.global_step % self.cfg.train.checkpoint_epochs_steps == 0:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model_' + str(self.global_step) + '.tar'))

            self.epoch += 1

        self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))
        logger.info(f'[TRAINER] Fitting has ended!')
