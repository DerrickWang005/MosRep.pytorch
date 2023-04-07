#!/usr/bin/env python
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import RoIAlign
from utils.misc import accuracy


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder_k = base_encoder(num_classes=dim, zero_init_residual=True)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, batch):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # parse batch
        im_q, im_k = batch
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # criterion & accuracy
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(logits, labels)[0]

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return loss, acc


class ConvProjector(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvProjector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features))

    def forward(self, x):
        """
            x: B x C x H x W
        """
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.projector(x)
        return x


class MosRep(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.2,
                 shift_enable=False, shift_pix=48, shift_beta=0.5):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MosRep, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.shift_enable = shift_enable
        self.shift_pix = shift_pix
        self.shift_beta = shift_beta

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder_k = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder_q.flatten = nn.Identity()
        self.encoder_k.flatten = nn.Identity()
        self.encoder_q.avgpool = nn.Identity()
        self.encoder_k.avgpool = nn.Identity()
        self.encoder_q.fc = nn.Identity()
        self.encoder_k.fc = nn.Identity()

        # create projector
        # dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.projector_q = ConvProjector(2048, dim)
        self.projector_k = ConvProjector(2048, dim)

        # create roi_align
        self.roi_align = RoIAlign(
            output_size=7,
            spatial_scale=1 / 32.0,
            sampling_ratio=-1,
            aligned=True)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _shift(self, mosaic, coord, enable=False, range=48., beta=0.5):
        if enable is False:
            area = torch.ones(coord.size(
                0), device=mosaic.device).repeat(mosaic.size(0))
            return mosaic, coord, area
        else:
            canvas = torch.zeros_like(mosaic)
            B, C, H, W = mosaic.size()
            h, w = H // 2, W // 2
            offset_y = int((np.random.beta(beta, beta) * 2. - 1.) * range)
            offset_x = int((np.random.beta(beta, beta) * 2. - 1.) * range)
            x1m, x1c = max(0, offset_x), max(0, -offset_x)
            y1m, y1c = max(0, offset_y), max(0, -offset_y)
            x2m, x2c = min(W, W + offset_x), min(W - offset_x, W)
            y2m, y2c = min(H, H + offset_y), min(H - offset_y, H)
            canvas[..., y1c:y2c, x1c:x2c] = mosaic[..., y1m:y2m, x1m:x2m]
            coord = torch.tensor([
                [x1c, y1c, w - offset_x - 1, h - offset_y - 1],
                [w - offset_x, y1c, x2c - 1, h - offset_y - 1],
                [x1c, h - offset_y, w - offset_x - 1, y2c - 1],
                [w - offset_x, h - offset_y, x2c - 1, y2c - 1],
            ], device=mosaic.device, dtype=torch.float32)
            area = (coord[:, 2] - coord[:, 0] + 1) * \
                   (coord[:, 3] - coord[:, 1] + 1)
            area = (area / (h * w)).sqrt().repeat(B)

            return canvas, coord, area

    @torch.no_grad()
    def _mosaic_ddp(self, x, batch_idx):
        """
        x: (b, 4, 3, h, w)
        batch_idx: (b, 4)
        """
        b, k, c, h, w = x.size()
        # shuffle
        ids_shuffle = [torch.arange(b)]
        for i in range(1, k):
            ids_shuffle.append(
                torch.arange(b).roll(i)
            )
        ids_shuffle = torch.stack(ids_shuffle, 1).to(x.device)
        x_shuffled = x.gather(
            dim=0,
            index=ids_shuffle.reshape(b, k, 1, 1, 1).repeat(1, 1, c, h, w)
        )
        mosaic_idx = batch_idx.gather(
            dim=0,
            index=ids_shuffle
        )
        # compose
        mosaic = rearrange(
            x_shuffled, 'b (k1 k2) c h w -> b c (k1 h) (k2 w)', k1=2, k2=2)
        mosaic_idx = mosaic_idx.flatten()
        # shift
        B, _, H, W = mosaic.size()
        coord = torch.tensor(
            [
                [0, 0, w - 1, h - 1],
                [w, 0, W - 1, h - 1],
                [0, h, w - 1, H - 1],
                [w, h, W - 1, H - 1],
            ],
            device=mosaic.device,
            dtype=torch.float32,
        )
        enable = random.random() < self.shift_enable
        mosaic, coord, area = self._shift(
            mosaic, coord, enable, self.shift_pix, self.shift_beta)
        coords = [coord] * B

        return mosaic, mosaic_idx, coords, area

    def forward(self, batch, multi_crop):
        # parse batch
        im_q, im_q_mini, im_k = batch
        batch_idx = torch.arange(
            im_q_mini.size(0),
            device=im_q_mini.device).unsqueeze(1).repeat(1, im_q_mini.size(1))
        # mosaic transform
        im_q_mini, batch_idx, rois, area = self._mosaic_ddp(im_q_mini, batch_idx)
        # compute query-single features
        q = self.encoder_q(im_q)  # queries: B, C, H, W
        q_sgl = self.projector_q(q)  # queries: B, C
        q_sgl = F.normalize(q_sgl, dim=-1)
        # compute query-mosaic features
        q_mos = self.encoder_q(im_q_mini)  # queries: B, C, H, W
        q_mos = self.roi_align(q_mos, rois)  # BK, C, H, W
        q_mos = self.projector_q(q_mos)  # queries: BK, C
        q_mos = F.normalize(q_mos, dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # compute key features
            k = self.encoder_k(im_k)  # keys: B, C, HxW
            k = self.projector_k(k)  # keys: BxC
            k = F.normalize(k, dim=-1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_gather = concat_all_gather(k)

        # compute logits: mosaic
        # positive logits: BK, B'
        pos_mos = torch.einsum("nc,mc->nm", [q_mos, k_gather])
        # negative logits: BK, N
        neg_mos = torch.einsum(
            "nc,ck->nk", [q_mos, self.queue.clone().detach()])
        # logits: BK, (B'+N)
        log_mos = torch.cat([pos_mos, neg_mos], dim=1) / self.T
        # labels: positive key indicators
        tgt_mos = batch_idx.clone() + im_q.size(0) * dist.get_rank()
        # criterion & accuracy
        loss_mos = F.cross_entropy(log_mos, tgt_mos, reduction="none")
        loss_mos = (loss_mos * area).mean()
        acc_mos = accuracy(log_mos, tgt_mos)[0]

        # compute logits: object
        # positive logits: B, B'
        pos_sgl = torch.einsum("nc,mc->nm", [q_sgl, k_gather])
        # negative logits: B, N
        neg_sgl = torch.einsum(
            "nc,ck->nk", [q_sgl, self.queue.clone().detach()])
        # logits: B, (B'+N)
        log_sgl = torch.cat([pos_sgl, neg_sgl], dim=1) / self.T
        # labels: positive key indicators
        tgt_sgl = torch.arange(log_sgl.shape[0], dtype=torch.long, device=log_sgl.device) +\
            im_q.size(0) * dist.get_rank()
        # criterion & accuracy
        loss_sgl = F.cross_entropy(log_sgl, tgt_sgl)
        acc_sgl = accuracy(log_sgl, tgt_sgl)[0]

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_gather)

        # return logits, labels
        return loss_sgl, loss_mos, acc_sgl, acc_mos


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
