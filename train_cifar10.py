# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "einops",
#   "wandb",
#   "tqdm",
#   "x-transformers",
#   "x-mlps-pytorch",
#   "vector-quantize-pytorch",
#   "ema-pytorch",
#   "torch-einops-utils",
#   "fire",
#   "accelerate"
# ]
# ///

import fire
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T

from einops.layers.torch import Rearrange
from accelerate import Accelerator

from tqdm import tqdm
import wandb

from ASAC import ASAC, PatchEmbedding, EMA_ASAC

# train

def main(
    use_asac: bool = False,
    use_ema_targets: bool = False,
    ema_decay: float = 0.999,
    eval_use_base: bool = False,
    cpu: bool = False,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    dim: int = 256,
    depth: int = 6,
    heads: int = 8,
    recon_loss_weight: float = 1.,
    commit_loss_weight: float = 1.,
    kl_div_loss: bool = True,
    project_name: str = 'asac-kl-div'
):
    accelerator = Accelerator(cpu = cpu, log_with = 'wandb')

    run_name = 'asac' if use_asac else 'baseline'
    if use_asac and kl_div_loss:
        run_name += '-kldiv'

    accelerator.init_trackers(
        project_name = project_name,
        init_kwargs = dict(wandb = dict(name = f'cifar10-{run_name}'))
    )

    transform = T.Compose([
        T.RandomCrop(32, padding = 4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainloader = DataLoader(datasets.CIFAR10('./data', train = True, download = True, transform = transform), batch_size = batch_size, shuffle = True, drop_last = True)
    testloader = DataLoader(datasets.CIFAR10('./data', train = False, download = True, transform = transform_test), batch_size = batch_size, shuffle = False)

    to_embedding = PatchEmbedding(dim = dim, patch_size = 4, channels = 3)

    model = ASAC(
        dim = dim,
        depth = depth,
        heads = heads,
        seq_len = 64,
        to_embedding = to_embedding,
        use_asac = use_asac,
        recon_loss_weight = recon_loss_weight,
        commit_loss_weight = commit_loss_weight,
        kl_div_loss = kl_div_loss
    )

    if use_ema_targets:
        model = EMA_ASAC(model, ema_decay = ema_decay)

    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(trainloader, desc = f'epoch {epoch+1}/{epochs}')

        for inputs, targets in pbar:
            optimizer.zero_grad()
            ret = model(inputs)
            outputs, aux_loss, (recon_loss, commit_loss) = ret.logits, ret.aux_loss, ret.aux_loss_breakdown

            loss = F.cross_entropy(outputs, targets)
            accelerator.backward(loss + aux_loss)
            optimizer.step()
            
            if use_ema_targets:
                accelerator.unwrap_model(model).update()

            acc = (outputs.argmax(dim = -1) == targets).float().mean()

            pbar.set_postfix(
                loss = float(loss),
                acc = float(acc),
                recon = float(recon_loss),
                commit = float(commit_loss)
            )

        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_loss, test_acc, total = 0., 0., 0
            for inputs, targets in testloader:
                batch = targets.shape[0]

                if use_ema_targets:
                    ret = model(inputs, use_ema = not eval_use_base)
                else:
                    ret = model(inputs)

                outputs = ret.logits
                
                test_loss += F.cross_entropy(outputs, targets).item() * batch
                test_acc += (outputs.argmax(dim = -1) == targets).float().sum().item()
                total += batch

        val_acc = test_acc / total
        val_loss = test_loss / total

        print(f'epoch {epoch+1}: val acc: {val_acc:.4f}, val loss: {val_loss:.4f}')
        accelerator.log(dict(val_loss = val_loss, val_acc = val_acc), step = epoch)

    accelerator.end_training()

if __name__ == '__main__':
    fire.Fire(main)
