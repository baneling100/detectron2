#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

"""
import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
"""
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from detectron2.config import get_cfg
from detectron2.evaluation import verify_results
from detectron2.projects.point_rend import add_pointrend_config

from custom_trainer import CustomTrainer
from custom_checkpointer import CustomDetectionCheckpointer
from custom_util import custom_argument_parser, custom_setup
"""
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config
"""

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    devices = xm.get_xla_supported_devices()
    assert devices, 'No devices of {} kind'.format('ANY')
    cfg.MODEL.DEVICE = devices[0]
    cfg.freeze()
    custom_setup(cfg, args)
    return cfg


def main(index, args):
    cfg = setup(args)

    if args.eval_only:
        model = CustomTrainer.build_model(cfg)
        CustomDetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = CustomTrainer.test(cfg, model)
        if xm.is_master_ordinal():
            verify_results(cfg, res)
        return res

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = custom_argument_parser().parse_args()
    print("Command Line Args:", args)
    xmp.spawn(
        main,
        nprocs=args.num_tpus,
        #num_machines=args.num_machines,
        #machine_rank=args.machine_rank,
        #dist_url=args.dist_url,
        args=(args,),
    )