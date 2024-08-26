#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler

import itertools
from typing import Optional


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will prepend a dimension, whilst ensuring it stays the same across one mini-batch.
    """

    def __init__(self, *args, input_dimension=None, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None
        self.mosaic = mosaic

    def __iter__(self):
        self.__set_input_dim()
        for batch in super().__iter__():
            yield [(self.input_dim, idx, self.mosaic) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


class TwoFrameSampler(Sampler):
    def __init__(self,dataset,batch,rank=0,world_size=1,seed: Optional[int] = 0):
        self.size = len(dataset)
        self.dataset = dataset
        self.video_label = self.getlabel()
        #for i in range(self.size):
        """
            if(i<215):#mot20!
                self.videolabel.append(1)
                continue
            elif(i<1607):
                self.videolabel.append(2)
                continue
            elif(i<2810):
                self.videolabel.append(3)
                continue
            elif(i<4468):
                self.videolabel.append(4)
                continue"""
        """
            if(i<703):#dancetrack!
                self.videolabel.append(1)
                continue
            elif(i<1906):
                self.videolabel.append(2)
                continue
            elif(i<3108):
                self.videolabel.append(3)
                continue
            elif(i<3991):
                self.videolabel.append(4)
                continue
            elif(i<5194):
                self.videolabel.append(5)
                continue
            elif(i<6397):
                self.videolabel.append(6)
                continue
            elif(i<8560):
                self.videolabel.append(7)
                continue
            elif(i<9143):
                self.videolabel.append(8)
                continue
            elif(i<10626):
                self.videolabel.append(9)
                continue
            elif(i<11389):
                self.videolabel.append(10)
                continue
            elif(i<11792):
                self.videolabel.append(11)
                continue
            elif(i<13055):
                self.videolabel.append(12)
                continue
            elif(i<13659):
                self.videolabel.append(13)
                continue
            elif(i<14462):
                self.videolabel.append(14)
                continue
            elif(i<15665):
                self.videolabel.append(15)
                continue
            elif(i<16907):
                self.videolabel.append(16)
                continue
            elif(i<18110):
                self.videolabel.append(17)
                continue
            elif(i<19313):
                self.videolabel.append(18)
                continue
            elif(i<20516):
                self.videolabel.append(19)
                continue
            elif(i<21719):
                self.videolabel.append(20)
                continue
            elif(i<22922):
                self.videolabel.append(21)
                continue
            elif(i<24126):
                self.videolabel.append(22)
                continue
            elif(i<25329):
                self.videolabel.append(23)
                continue
            elif(i<25951):
                self.videolabel.append(24)
                continue
            elif(i<27154):
                self.videolabel.append(25)
                continue
            elif(i<28357):
                self.videolabel.append(26)
                continue
            elif(i<29559):
                self.videolabel.append(27)
                continue
            elif(i<30762):
                self.videolabel.append(28)
                continue
            elif(i<32165):
                self.videolabel.append(29)
                continue
            elif(i<33368):
                self.videolabel.append(30)
                continue
            elif(i<34571):
                self.videolabel.append(31)
                continue
            elif(i<35374):
                self.videolabel.append(32)
                continue
            elif(i<36575):
                self.videolabel.append(33)
                continue
            elif(i<37178):
                self.videolabel.append(34)
                continue
            elif(i<37781):
                self.videolabel.append(35)
                continue
            elif(i<38384):
                self.videolabel.append(36)
                continue
            elif(i<39387):
                self.videolabel.append(37)
                continue
            elif(i<39990):
                self.videolabel.append(38)
                continue
            elif(i<41193):
                self.videolabel.append(39)
                continue
            elif(i<41796):
                self.videolabel.append(40)
                continue"""
        self.batch_size = batch
        assert self.size > 0
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size
    def get_iter(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        it = torch.randperm(self.size,generator=g).tolist()
        import random
        #add = [random.randint(1,5) for _ in range(self.size)]
        add = [1 for _ in range(self.size)]
        it2 = [it[i]+add[i] for i in range(self.size)]
        for i in range(self.size-1,-1,-1):
            if(it2[i]>=self.size):
                it.pop(i)
                it2.pop(i)
                continue
            video_id1 = self.video_label[it[i]]
            video_id2 = self.video_label[it2[i]]
            if(video_id1!=video_id2):
                it.pop(i)
                it2.pop(i)
        gen = [(it[i],it2[i]) for i in range(len(it))]
        gen = torch.IntTensor(gen)
        return iter(gen)
    def getlabel(self):
        import os
        path = "your_dataset_path" #need to modify.
        seqs = os.listdir(path)
        video_id = 0
        out = []
        for seq in sorted(seqs):
            img_path = os.path.join(path,seq)
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            video_id +=1
            for i in range(num_images):
                out.append(video_id)
        return out
    def __iter__(self):
        start = self._rank#first index
        
        while True:
            samp = self.get_iter()
            yield from itertools.islice(
            samp,start,None,self._world_size
            )



    def __len__(self):
        return self.size // self._world_size
    

