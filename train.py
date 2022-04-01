#!/usr/bin/env python3
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from torchmetrics.image.psnr import PeakSignalNoiseRatio

import utils
from fid import FID
from collage import Collage
from model import Model

PROJECT_NAME = 'auto-encoder'  # TODO: Change project name
PROJECT_HPARAM_DISP = ['dataset',]  # TODO: Add hyper-params to display in project name
RUN_HPARAM_DISP = ['lr',]  # TODO: Add hyper-params to display in run names

def main(args, proj_name, run_name):
    # ensures that weight initializations are all the same
    pl.seed_everything(args.seed)
    gpu_num = (args.gpus if torch.cuda.is_available() else 0)
    logger = None
    callbacks = []
    metrics = {'PSNR': PeakSignalNoiseRatio()}
    if not args.fast_dev_run:
        logger = pl_loggers.WandbLogger(name=run_name, project=proj_name, entity='gip')
        logger.log_hyperparams(args)
        if args.img_size[0] == 3:
            metrics['FID'] = FID()
        callbacks += [Collage(Model.collage_methods())]
    metrics = torchmetrics.MetricCollection(metrics)
    model = Model(metrics, **vars(args))
    model.prepare_data()
    trainer = pl.Trainer.from_argparse_args(
        args, default_root_dir='logs', logger=logger,
        callbacks=callbacks, strategy='ddp' if gpu_num != 1 else None
    )
    trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Model.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args = utils.argparse_str2list(args)
    proj_name = PROJECT_NAME + utils.hparams2desc(parser, args, PROJECT_HPARAM_DISP, verbose='v')
    run_name = PROJECT_NAME + utils.hparams2desc(parser, args, RUN_HPARAM_DISP, verbose='vvv')
    main(args, proj_name, run_name)
