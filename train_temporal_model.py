from models.temporal_model_branched import TemporalModelBranched
from datasets.branched_temporal_dataset import BranchedTemporalDataset

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import datetime

from machines import machines

import socket

import random

import logging

def train(rank, size):
    batch_size = 32


    train_dataset = BranchedTemporalDataset()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=size,
                                                                    rank=rank,
                                                                    shuffle=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)

    # bsz = batch_size / float(size)
    # partition_sizes = [1.0 / size for _ in range(size)]
    # partition = DataPartitioner(train_dataset, partition_sizes)
    # partition = partition.use(dist.get_rank())
    #
    # train_set = torch.utils.data.DataLoader(partition,
    #                                      batch_size=bsz,
    #                                      shuffle=True)
    # print("here3")
    model = TemporalModelBranched()

    epochs = 10000
    logging.basicConfig(filename="/s/chopin/k/grad/sarmst/CR/logs/" + socket.gethostname() + ".log", level=logging.INFO)
    logging.info("Begin Training")
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            # print("here4")
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # print("here5")
            model.forward()                   # compute fake images: G(A)
            # update D
            # print("here6")
            model.set_requires_grad(model.netD, True)  # enable backprop for D
            # print("here7")
            model.optimizer_D.zero_grad()     # set D's gradients to zero
            # print("here8")
            model.backward_D()                # calculate gradients for D
            # print("here9")
            average_gradients(model.netD)
            # print("here10")
            model.optimizer_D.step()          # update D's weights
            # update G
            # print("here11")
            model.set_requires_grad(model.netD, False)  # D requires no gradients when optimizing G
            # print("here12")
            model.optimizer_G.zero_grad()        # set G's gradients to zero
            # print("here13")
            model.backward_G()                   # calculate graidents for G
            # print("here14")
            average_gradients(model.netG)
            # print("here15")
            model.optimizer_G.step()
        logging.info('Rank: ' + str(dist.get_rank()) + ', Epoch: ' + str(epoch) + ', Loss: ' + str(model.loss_G.item()))
        if rank == 0:
            model.save_networks(epoch)
        model.epoch_count += 1

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'lattice-1'
    os.environ['MASTER_PORT'] = '13131'
    # store = torch.distributed.FileStore("/s/chopin/k/grad/sarmst/CR/store/storeFile", size)

    dist.init_process_group(backend, timeout=datetime.timedelta(0, 180000), init_method='file:///s/chopin/k/grad/sarmst/CR/checkpoints/sharedFile', rank=rank, world_size=size)

    fn(rank, size)


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    size = len(machines)
    rank = machines.index(str(socket.gethostname()))

    print("Size is " + str(size) + ", Rank is " + str(rank))

    processes = []
    # for rank in range(size):
    p = Process(target=init_process, args=(rank, size, train))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
