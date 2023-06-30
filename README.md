# training-methods-tutorial

本文档提供了使用不同模式进行CIFAR-10分类任务模型训练的详细步骤，包括单机单卡、单机多卡和多机多卡等训练方法。文档将介绍如何使用原生**Pytorch**的纯**GPU**、**DataParallel**和**DistributedDataParallel**等训练方式和**DeepSpeed**的大模型分布式训练。

## 1. 训练任务简介
本训练任务使用CIFAR-10数据集，该数据集包含10个类别的图像，包含6000张32x32彩色图像，并旨在训练一个模型，对这些图像进行分类，下文基于该例子介绍各种训练方式。

## 2. 训练方法汇总
### 2.1 原生训练（纯CPU）
1. 适用于在没有GPU的情况下进行模型训练。
2. 训练过程在CPU上运行，速度较慢。
3. 参考代码[ 01_cifar10_cpu.py](./01_cifar10_cpu.py)

### 2.2 单机单卡训练（纯GPU）
1. 适用于在具有单个GPU的机器上进行模型训练。
2. 将模型和数据移动到GPU上进行训练，加速训练过程。
3. 参考代码[ 02_cifar10_gpu.py](./02_cifar10_gpu.py)

#### 2.2.1 训练步骤（纯GPU）    
1. **确保GPU可用：** 使用 `torch.cuda.is_available()` 获取 GPU device；
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

2. **模型迁移到GPU上：** 使用 `model.to(device)`；
```python
model.to(device)
```

3. **数据加载并迁移到GPU上：** 使用 `data.to(device)`；
```python
trainset, testset, trainloader, testloader, trainsampler = get_dataset(mode="gpu", batch_size=batch_size)

...
for epoch in range(epochs):
    for inputs, labels in trainloader:
        # 迁移到GPU上
        inputs, labels = inputs.to(device), labels.to(device)
        ...
        loss.backward()
        optimizer.step()
```

4. **模型保存：** 使用 `model.state_dict()` 保存即可；
```python
if step % save_step_interval == 0:
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"cifar_net_step_{step}.pth")
    torch.save({'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, save_file)
```

5. **执行方式：**
```
python train.py
```

### 2.3 单机多卡训练
1. 适用于在具有多个GPU的机器上进行模型训练。
2. 可使用 `DataParallel`/`DistributedDataParallel` 将模型复制到多个GPU上，实现并行训练。
3. DataParallel特点：参考代码[ 03_cifar10_gpu_dataparallel.py](./03_cifar10_gpu_dataparallel.py)
    - 单进程，效率慢；
    - 不支持多机多卡训练；
    - 不支持模型并行；

4. DistributedDataParallel：参考代码[ 04_cifar10_gpu_distributeddataparallel.py](./04_cifar10_gpu_distributeddataparallel.py)

#### 2.3.1 训练步骤（**DataParallel**）

1. **确保GPU可用：** 使用 `torch.cuda.is_available()` 获取 GPU device；
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

2. **包装模型：** 使用 `torch.nn.DataParallel` 包装 model.to(device)；
```python
model = nn.DataParallel(model.to(device), device_ids=[0, 1])
```

3. **数据加载并迁移到GPU上：** 使用 `data.to(device)`；这里的batch_size应该是每个`GPU的总和`；
```python
trainset, testset, trainloader, testloader, trainsampler = get_dataset(mode="gpu", batch_size=batch_size)

...
for epoch in range(epochs):
    for inputs, labels in trainloader:
        ####### move to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        ...
        loss.backward()
        optimizer.step()
```

4. **模型保存：** 使用 `model.module.state_dict()` 保存即可；
```python
if step % save_step_interval == 0:
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"cifar_net_step_{step}.pth")
    torch.save({'epoch': epoch_index,
                ####### 3. use model.module.state_dict() instead of model.state_dict()
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, save_file)
```

5. **执行方式：**
```
python train.py
```

#### 2.3.2 训练步骤（**DistributedDataParallel**）

1. **初始化1：** 使用 `init_process_group` 初始化；
```python
n_gpus = 2
####### 1. init_process_group
torch.distributed.init_process_group("nccl", world_size=n_gpus)
```
2. **初始化2：** 获取当前进程设备`device`，并使用 `set_device` 设置进程设备；

```python
####### 2. set local device
local_rank = torch.distributed.get_rank()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(local_rank)
```

3. **包装模型：** 使用 `torch.nn.parallel.DistributedDataParallel` 包装 model.to(device)，可指定设备device_ids；
```python
model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank])   
```

4. **数据加载：** 使用 `torch.utils.data.distributed.DistributedSampler` 设置分布式采样；有了sampler，`shuffle=False`即可；这里的batch_size`不是总和`，应该是`每个GPU的batch_size`；
```python
# from get_dataset func
trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, rank=local_rank)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=trainsampler)
```

5. **数据打乱：** 使用 `trainsampler.set_epoch` 打乱数据，否则每个GPU分到数据都一样；同样需要迁移到device上；
```python
# from train func
for epoch in range(epochs):
    ####### 2. random dataset in every rank
    trainsampler.set_epoch(epoch_index)
    for inputs, labels in trainloader:
        ####### move to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        ...
        loss.backward()
        optimizer.step()
```

6. **模型保存：** 只在 `local_rank=0` 保存模型，且使用 `model.module.state_dict()` 保存；
```python
if step % save_step_interval == 0 and local_rank == 0:
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"cifar_net_step_{step}.pth")
    torch.save({'epoch': epoch_index,
                ####### 3. use model.module.state_dict() instead of model.state_dict()
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, save_file)
```

7. **结束处理：** 使用 `destroy_process_group` 结束分布式训练进程；
```python
torch.distributed.destroy_process_group()
```

8. **执行方式：** 使用 `torch.distributed.launch` 启动多个进程，来分别启动多卡训练任务；使用 `CUDA_VISIBLE_DEVICES=0,1` 指定设备；
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 04_cifar10_gpu_distributeddataparallel.py
```

### 2.4 多机多卡训练
1. 配置多台机器，并确保它们之间可以互相访问。
2. 在每台机器上准备数据集并配置相同的模型。
3. 训练脚本支持多机多卡训练。
4. DistributedDataParallel：参考代码[ 04_cifar10_gpu_distributeddataparallel.py](./04_cifar10_gpu_distributeddataparallel.py)
5. DeepSpeed：参考代码[ 05_cifar10_gpu_deepspeed.py](./05_cifar10_gpu_deepspeed.py)

#### 2.4.1 训练步骤（**DistributedDataParallel**）

1. **修改代码：** 直接使用上面单机多卡的代码即可[ 04_cifar10_gpu_distributeddataparallel.py](./04_cifar10_gpu_distributeddataparallel.py)

2. **执行方式：** 在每台多卡机器的运行下面训练脚本；`NUM_GPUS` 指定每台机器上要使用的GPU数量；`nnodes` 指定节点数量；`node_rank` 指定当前节点rank；`master_addr` 指定主节点IP； 指定主节点的端口号；下面以两个节点为例：(例如主节点IP为10.38.234.187，端口为29500)。

```
# 在第1个机器节点上执行：
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=2 --node_rank=0 --master_addr=10.38.234.187 --master_port=29500 04_cifar10_gpu_distributeddataparallel.py

# 在第2个机器节点上执行：
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=2 --node_rank=1 --master_addr=10.38.234.187 --master_port=29500 04_cifar10_gpu_distributeddataparallel.py
```

#### 2.4.2 训练步骤（**DeepSpeed**）

1. **初始化1：** 使用 `init_process_group` 初始化；
```python
####### 1. init deepspeed
args = get_args()
deepspeed.init_distributed(dist_backend=args.backend, dist_init_required=True)
```

2. **初始化2：** 获取当前进程设备`device`，并使用 `set_device` 设置进程设备；

```python
####### 2. set local device
args.local_rank = torch.distributed.get_rank()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
```

3. **数据加载：** 使用 `get_dataset` trainset；
```python
trainset, testset, trainloader, _, _ = get_dataset(mode=mode, batch_size=batch_size)
```

4. **包装模型：** 使用 `deepspeed.initialize` 包装 model、parameters和trainset完成初始化；
```python
####### 1. model && data move to deepspeed
parameters = [p for p in model.parameters() if p.requires_grad]
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=trainset)
fp16 = model_engine.fp16_enabled()
print(f'fp16={fp16}')
```

5. **训练处理：** 需要将数据迁移到对应的GPU device上；
```python
# from train func
for epoch_index in range(start_epoch, start_epoch+epochs+1):
    ema_loss = 0.0
    for batch_index, data in enumerate(trainloader):
        step = num_batches*(epoch_index) + batch_index + 1
        ####### 2. data move to model_engine.local_rank
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        ####### 3. use model_engine.backward and model_engine.step
        model_engine.backward(loss)
        model_engine.step()
        ema_loss = 0.9*ema_loss + 0.1*loss
```

6. **模型保存：** 只在 `local_rank=0` 保存模型，且使用 `model.state_dict()` 保存；
```python
if step % save_step_interval == 0 and local_rank == 0:
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"cifar_net_step_{step}.pth")
    torch.save({'epoch': epoch_index,
                ####### 3. use model.module.state_dict() instead of model.state_dict()
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, save_file)
```

7. **执行方式：** 使用 `torch.distributed.launch` 启动多个进程，来分别启动多卡训练任务；使用 `CUDA_VISIBLE_DEVICES=0,1` 指定设备；
```shell
export CUDA_VISIBLE_DEVICES=0,1
deepspeed cifar10_deepspeed.py --deepspeed_config ds_config.json
```
