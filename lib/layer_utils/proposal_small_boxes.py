
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox_intersect import bbox_intersect


def proposal_small_boxes(rois, gt_smboxes):

    intersect = bbox_intersect(
      np.ascontiguousarray(rois[:, 1:5], dtype=np.float),
      np.ascontiguousarray(gt_smboxes[:, 1:5], dtype=np.float))

    # ind = np.where(intersect >= 0.8)

    overlap = []
    max_length = 0
    for i in range(rois.shape[0]):
        ind = np.where(intersect[i,:] >= 0.5)[0]
        if len(ind)>1:
            print(ind)

        if len(ind) == 0:
            overlap.append([[0,0]])
            if max_length < 1:
                max_length = 1
            continue

        ll = []
        for k in range(len(ind)-1):
            for l in range(k+1,len(ind)):
                ll.append([ind[k],ind[l]])

        if len(ll) > max_length:
            max_length = len(ll)
        overlap.append(ll)

    for i in range(rois.shape[0]):
        pad = max_length - len(overlap[i])
        overlap[i].extend([[0,0]]*pad)

    overlap = np.array(overlap)
    return overlap








