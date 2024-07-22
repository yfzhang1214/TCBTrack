# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir
 
class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.with_reid = True            # ReID

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (800, 1440)
        self.random_size = (18, 32)
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "test.json"

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = 20
        self.warmup_lr = 0
        #self.basic_lr_per_img = 0.002 / 64.0
        self.basic_lr_per_img = 0.001 /16.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 1
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 500
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (800, 1440)
        self.test_conf = 0.1
        self.nmsthre = 0.7
    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(),"dancetrack"),
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229,0.224,0.225),#imagenet的means和std
                max_labels=500,
            ),#dataset.annotations[0][0]:[obj_num,6]:tlbr+classid+trackid
        )
        total_ids = dataset.nID # need to check: ids start with 0
        
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size//dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        settings = {'total_ids': total_ids}
        return train_loader, settings
    
    def get_data_loader2(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            TwoFrameSampler,
            InfiniteSampler
        )
        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(),"dancetrack"),
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229,0.224,0.225),#imagenet的means和std
                max_labels=500,
            ),#dataset.annotations[0][0]:[obj_num,6]:tlbr+classid+trackid
        )
        total_ids = dataset.nID # need to check: ids start with 0
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size//dist.get_world_size()

        sampler = TwoFrameSampler(self.dataset,batch = batch_size,seed=self.seed if self.seed else 0)
        #sampler = InfiniteSampler(
        #    len(self.dataset), seed=self.seed if self.seed else 0
        #)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=False,
        )
        #count = 0
        #for i in batch_sampler:
        #    count+=1
        #    print(count)
        #    print(i)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory":True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        settings = {'total_ids':total_ids}
        return train_loader, settings
        #print(self.dataset[2650][2][-1].split(".")[-1])



    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform
        #testdev=True
        if testdev:
            valdataset = MOTDataset(
                data_dir=os.path.join(get_yolox_datadir(), "dancetrack"),
                json_file=self.test_ann,
                img_size=self.test_size,
                name='test',
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        else:
            valdataset = MOTDataset(
                data_dir=os.path.join(get_yolox_datadir(), "dancetrack"),
                json_file=self.val_ann,
                img_size=self.test_size,
                name='val',
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator



