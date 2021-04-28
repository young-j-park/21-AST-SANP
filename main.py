
import os
from time import time
import math
from pprint import pprint

import numpy as np
import torch
from torch.optim import Adam

from model.neural_process import NP
from model.attentive_np import ANP
from utility.pytorch_utils import tensor, zeros, to_np, ones_like
from data_loader import get_dataset
from config import DEVICE, EXPERIMENT_ID, RESULT_DIR


def grid(dx, dy, scale):
    x = np.linspace(0, dx-1, dx) / (dx-1) * scale*2 - scale
    y = np.linspace(0, dy-1, dy) / (dy-1) * scale*2 - scale
    xm, ym = np.meshgrid(x, y)
    return np.stack((ym, xm), -1).reshape((1, -1, 2))


def run(
        model_name='anp',  # choices = ['anp', 'np']
        dataset_name='apollo.npy',
        mask_fname='apollo_train_mask',
        window_size=100,
        sample_size=100,  # set -1 to run vanilla neural processes
        sample_scale_sq=31.25,
        emb_dim=512,
        eps=1e-3,
        fix_eps=150,
        mask_size=5,

        use_rotation_aug=True,
        use_scaling_aug=True,

        learning_rate=1e-4,
        batch_size=512,
        max_epoch=500,
        epoch_split=10,

        # FOR TRAIN
        model_path='',  # model path
        recon_nodata=False,  # reconstruct on no-data gaps

        # FOR RECONSTRUCTION
        # model_path='anp_{EXPERIMENT_ID}.pth',  # model path
        # recon_nodata=True,  # reconstruct on no-data gaps
):
    np.random.seed(7)
    torch.manual_seed(7)

    print(f'Experiment ID: {EXPERIMENT_ID}')
    print(DEVICE)
    if window_size % 2 == 0:
        window_size += 1
    assert window_size > mask_size
    assert sample_size < window_size**2
    center_idx = window_size ** 2 // 2  # index of center pixel

    # loaders
    train_loader, valid_loader, test_loader = get_dataset(
        dataset_name=dataset_name,
        window_size=window_size,
        batch_size=batch_size,
        mask_fname=mask_fname,
        mask_size=mask_size,
        epoch_split=epoch_split,
        recon_nodata=recon_nodata,
    )

    # models
    if model_name == 'np':
        model = NP(
            x_dim=2, y_dim=1, emb_dim=emb_dim,
            dist='Gaussian', stochastic=False,
        )
    elif model_name == 'anp':
        model = ANP(
            x_dim=2, y_dim=1, emb_dim=emb_dim,
            dist='Gaussian', stochastic=False,
        )
    else:
        raise NotImplementedError

    # load model
    if model_path:
        model_path = os.path.join(RESULT_DIR, model_path)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # initialize model and optimizer
    if recon_nodata:
        max_epoch = 1
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate)

    # initialize variables
    x_grid = tensor(grid(window_size, window_size, scale=2))
    p_grid = tensor(grid(window_size, window_size, scale=(window_size-1)/2))
    p_grid = torch.exp(-1/sample_scale_sq * (p_grid[0, :, 0]**2 + p_grid[0, :, 1]**2))

    best_valid_loss = np.inf
    if recon_nodata:
        loaders = {'test': test_loader}
    else:
        loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    for epoch in range(max_epoch):
        for run_type, loader in loaders.items():
            # initialize variables
            if run_type == 'train':
                model.train()
            else:
                model.eval()

            losses = []
            y_true = []
            y_pred = []
            y_sig = []
            idx = []
            tic = time()
            for i, batch_data in enumerate(loader):
                y_context, context_mask, target_value, idx0, idx1 = batch_data
                bs = y_context.size(0)
                x_context = x_grid.expand(bs, -1, -1)

                context_mask[:, center_idx:center_idx + 1] = 0  # mask a center
                non_context = ~context_mask

                if 0 < sample_size:
                    if run_type == 'train':
                        sample_idx = torch.multinomial(p_grid, sample_size, replacement=False)
                        x_context = x_context[:, sample_idx]
                        y_context = y_context[:, sample_idx]
                        context_mask = context_mask[:, sample_idx]
                        non_context = non_context[:, sample_idx]
                    else:
                        sample_idx = []
                        for ib in range(bs):
                            prob = p_grid.clone()
                            prob[non_context[ib]] = 0
                            sample_idx.append(torch.topk(prob, sample_size)[1])
                        sample_idx = torch.cat(sample_idx)
                        batch_idx = torch.arange(bs).unsqueeze(-1).expand(-1, sample_size).flatten()
                        x_context = x_context[batch_idx, sample_idx].view(bs, sample_size, -1)
                        y_context = y_context[batch_idx, sample_idx].view(bs, sample_size, -1)
                        context_mask = context_mask[batch_idx, sample_idx].view(bs, sample_size)
                        non_context = non_context[batch_idx, sample_idx].view(bs, sample_size)

                # scale
                ml = context_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
                y_context[non_context] = 0.0
                mean = y_context.sum(dim=1, keepdim=True) / ml
                scale = (y_context - mean)**2
                scale[non_context] = 0.0
                scale = torch.sqrt(scale.sum(dim=1, keepdim=True) / ml)
                y_context = (y_context - mean) / scale

                # augment
                if run_type == 'train':
                    if use_rotation_aug:
                        theta = torch.rand(
                            bs, 1, 1, dtype=torch.float32, device=DEVICE
                        ) * (math.pi * 2)
                        cth = torch.cos(theta)
                        sth = torch.sin(theta)
                        x_context = torch.cat(
                            (x_context[:, :, 0:1] * cth - x_context[:, :, 1:2] * sth,
                             x_context[:, :, 0:1] * sth + x_context[:, :, 1:2] * cth),
                            dim=-1
                        )
                    if use_scaling_aug:
                        y_scale = torch.rand(
                            bs, 1, 1, dtype=torch.float32, device=DEVICE
                        ) + 0.5
                        y_context *= y_scale
                        scale *= y_scale

                # target value
                x_center = zeros(bs, 1, 2)
                y_target = model(
                    x_context, y_context, context_mask, non_context, x_center
                )
                mu, logvar = torch.chunk(y_target, 2, dim=-1)
                if epoch <= fix_eps:
                    sigma = eps * ones_like(logvar)
                else:
                    sigma = eps + torch.exp(0.5 * logvar)

                # rescale
                mu = mu * scale + mean
                sigma *= scale

                # compute loss and update
                loss = torch.mean(
                    0.5 * ((target_value - mu) / sigma) ** 2 + torch.log(sigma),
                )
                if run_type == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())
                y_true.append(to_np(target_value))
                y_pred.append(to_np(mu))
                y_sig.append(to_np(sigma))
                idx.append(np.concatenate((to_np(idx0), to_np(idx1)), 1))

            # report results
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            y_sig = np.concatenate(y_sig)
            idx = np.concatenate(idx)

            # save sample results
            if recon_nodata:
                fname = os.path.join(
                    RESULT_DIR, f'recon_{EXPERIMENT_ID}.npz'
                )
                np.savez(
                    fname, y_pred=y_pred.flatten(), y_sig=y_sig.flatten(), idx=idx
                )
            else:
                l1_err = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                loss = np.mean(losses)

                if run_type == 'valid' and loss < best_valid_loss:
                    print('Best !!')
                    best_valid_loss = loss

                    # save model
                    fname = os.path.join(
                        RESULT_DIR, f'{model_name}_{EXPERIMENT_ID}.pth'
                    )
                    torch.save(model.state_dict(), fname)

                report_dict = {
                    'epoch': epoch,
                    f'{run_type}__loss': float(loss.item()),
                    f'{run_type}__l1err': float(l1_err),
                    f'{run_type}__rmse': float(rmse),
                    f'{run_type}__epochtime': float(time() - tic),
                }
                pprint(report_dict)


if __name__ == '__main__':
    run()
