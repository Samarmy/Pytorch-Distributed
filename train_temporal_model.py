from models.temporal_model_branched import TemporalModelBranched
from datasets.branched_temporal_dataset import BranchedTemporalDataset

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from machines import machines

def train(rank, size):

    batch_size = 1

    train_dataset = BranchedTemporalDataset()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=size,
                                                                    rank=rank,
                                                                    shuffle=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=size,
                                               sampler=train_sampler)

    model = TemporalModelBranched()

    epochs = 10000
    print("Begin Training")
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            model.set_input(data)         # unpack data from dataset and apply preprocessing

            model.forward()                   # compute fake images: G(A)
            # update D
            model.set_requires_grad(model.netD, True)  # enable backprop for D
            model.optimizer_D.zero_grad()     # set D's gradients to zero
            model.backward_D()                # calculate gradients for D
            average_gradients(model.netD)
            model.optimizer_D.step()          # update D's weights
            # update G
            model.set_requires_grad(model.netD, False)  # D requires no gradients when optimizing G
            model.optimizer_G.zero_grad()        # set G's gradients to zero
            model.backward_G()                   # calculate graidents for G
            average_gradients(model.netG)
            model.optimizer_G.step()

            print('Rank ', dist.get_rank(), ', epoch ', epoch)

        if rank == 0:
            model.save_networks(epoch)
        model.update_learning_rate()
        model.epoch_count += 1


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'kenai'
    os.environ['MASTER_PORT'] = '13131'
    # store = torch.distributed.FileStore("/s/chopin/k/grad/sarmst/CR/store", size)
    dist.init_process_group(backend, rank=rank, world_size=size)
    # dist.init_process_group(backend, init_method='tcp://kenai:13131', rank=rank, world_size=size)
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
    import argparse

    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='Pytorch Distributed')
    parser.add_argument("--size")
    parser.add_argument("--rank")
    args = parser.parse_args()
    size = int(args.size)
    rank = machines[args.rank]
    print("Size is " + str(size))
    print("Rank is " + str(rank))

    processes = []
    # for rank in range(size):
    p = Process(target=init_process, args=(rank, size, train))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
