import os
import time
import tempfile
import numpy as np
import torch

import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils.utils import init_dataset_kwargs 
from torch_utils.utils import compose_model

def train_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    validation_set_kwargs   = {},       # Options for validation set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    model_kwargs            = {},       # Options for training model.
    optimiser_kwargs        = {},       # Options for generator optimizer.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    max_epochs              = 100,      # Maximum training epoch.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    ):

    # Initialize.
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    validation_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs) # subclass of training.dataset.Dataset
    validation_set_sampler = misc.InfiniteSampler(dataset=validation_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    validation_set_iterator = iter(torch.utils.data.DataLoader(dataset=validation_set, sampler=validation_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    
    if rank == 0:
        print()
        print('[Training] Num images: ', len(training_set))
        print('[Training] Image shape:', training_set.image_shape)
        print('[Training] Label shape:', training_set.label_shape)
        print('[Validation] Num images: ', len(validation_set))
        print('[Validation] Image shape:', validation_set.image_shape)
        print('[Validation] Label shape:', validation_set.label_shape)
        print()
    
    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    
    model = compose_model(**model_kwargs).train().requires_grad_(False).to(device)

    if rank == 0:
        print(model)

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [model]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Training Setting.
    if rank == 0:
        print('Setting up training optimiser...')
    optimiser = dnnlib.util.construct_class_by_name(module.parameters(), **optimiser_kwargs) # subclass of torch.optim.Optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # Train.
    if rank == 0:
        print(f'Start Training...')
        print()

    for tick in range(max_epochs * len(training_set) // batch_gpu + 1):
        # Fetch training data.
        gt_img, gt_label = next(training_set_iterator)
        gt_img = (gt_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)

        # Accumulate gradients.
        optimiser.zero_grad(set_to_none=True)
        model.requires_grad_(True)
        outputs = model(gt_img)
        loss = criterion(outputs, gt_label)
        loss.backward()
        model.requires_grad_(False)

        # Update weights.
        params = [param for param in model.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)
        optimiser.step()

def subprocess_fn(rank, c, temp_dir):
    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    train_loop(rank=rank, **c)


if __name__ == '__main__':
    # configs.
    NUM_GPU         = 2
    SEED            = 1234
    DATASET_PATH    = ''

    c = dnnlib.EasyDict(num_gpus=2)
    c.training_set_kwargs, train_dataset_name = init_dataset_kwargs(data=f'{DATASET_PATH}/train')
    c.validation_set_kwargs, val_dataset_name = init_dataset_kwargs(data=f'{DATASET_PATH}/val')
    c.random_seed = c.training_set_kwargs.random_seed = c.validation_set_kwargs.random_seed = SEED

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)
