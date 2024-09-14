import numpy as np
from scipy.cluster.vq import kmeans
import torch
import random


def auto_kmeans_anchors(datasets, cfg):
    shapes = []
    bboxes = []
    for dataset in datasets:
        for data in dataset.dataset_infos:
            bboxes.append(data['label'])
            shapes.append(data['wh_original'])
    WH = np.asarray(cfg.TRAIN.IMG_SIZE[::-1])
    shapes = np.asarray(shapes)
    shapes = WH / shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    ratios = shapes*scale
    # wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, bboxes)])).float()

    wh0 = []
    for bbox, ratio in zip(bboxes, ratios):
        _bbox = np.asarray(bbox)[:, 1:]
        wh = _bbox[:, 2:] - _bbox[:, :2]
        wh *= ratio
        wh0.append(wh)
    wh0 = np.concatenate(wh0)
    num_anchors = len(cfg.TRAIN.ANCHORS)

    s = wh0.std(0)  # sigmas for whitening
    k, dist = kmeans(wh0 / s, k_or_guess=num_anchors, iter=30)  # points, mean distance
    k *= s
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered

    npr = np.random
    f, sh, mp, s = anchor_fitness(k, wh0), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    # Evolve
    for _ in range(1000):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg, wh0)
        if fg > f:
            f, k = fg, kg.copy()
    k = k[np.argsort(k.prod(1))][::-1]
    anchor = []
    for ki in k:
        anchor.append([round(ki[0]), round(ki[1])])
    return anchor


def metric(k, wh):  # compute metrics
    r = wh[:, None] / k[None]
    x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
    # x = wh_iou(wh, torch.tensor(k))  # iou metric
    return x, x.max(1)[0]  # x, best_x


def anchor_fitness(k, wh, thr=0.25):  # mutation fitness
    _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
    return (best * (best > thr).float()).mean()  # fitness
