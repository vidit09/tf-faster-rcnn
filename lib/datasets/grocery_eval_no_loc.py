# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import os
import re
import pickle
import numpy as np
import cv2

from model.nms_wrapper import nms
def parse_rec(filename):
    """ Parse a INRIA-Person annotation file """
    objects = []
    with open(filename) as f:
        data = f.read()
    objs = re.findall('\d+ \d+ \d+ \d+', data)

    for ix, obj in enumerate(objs):
        obj_struct = {}
        coor = re.findall('\d+', obj)
        obj_struct['bbox'] = [int(coor[0]),
                              int(coor[1]),
                              int(coor[2]),
                              int(coor[3])]
        obj_struct['name'] = 'object'
        obj_struct['difficult'] = 0
        objects.append(obj_struct)
    return objects

def parse_rec_all(filename):
    """ Parse a INRIA-Person annotation file """
#    print(filename)
    objects = []
    with open(filename) as f:
        data = f.read()
    objs = re.findall('\d+ \d+ \d+ \d+ \d+', data)

    for ix, obj in enumerate(objs):
        obj_struct = {}
        coor = obj.split()
        obj_struct['bbox'] = [int(coor[0]),
                              int(coor[1]),
                              int(coor[2]),
                              int(coor[3])]
 #       print(coor[4])
        obj_struct['name'] = 'object_'+coor[4]
        obj_struct['difficult'] = 0
        objects.append(obj_struct)
    return objects


def grocery_ap(rec, prec, use_07_metric=False):
    """ ap = grocery_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
      #  for i in range(mpre.size - 1, 0, -1):
      #      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def grocery_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = grocery_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec_all(annopath.format(imagename))
            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print( 'Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    if len(lines)==0:
        return 0,0,0
    
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
#    print(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = grocery_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def grocery_eval_eccv_14(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             keep = 5,
             use_07_metric=False):
    """rec, prec, ap = grocery_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec_all(annopath.format(imagename))
            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print( 'Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0

   
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    ids = np.array([x[0] for x in splitlines])
#    print(ids)
    if len(ids) == 0:
        return 0,0,0
    
    
    total = []
    # for imagename in imagenames:
    R = recs[classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    gt = np.array([x['name'] for x in R])
    det = [False] * len(R)
    npos = sum(~difficult)
    R = {'bbox': bbox,
         'difficult': difficult,
         'det': det,
         'gt':gt}

    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

#    dets = np.hstack((BB,confidence[:,np.newaxis])).astype(np.float32, copy=False)
#    keep = nms(dets,0.2)
   
#    confidence = confidence[keep]
#    BB = BB[keep,:]
#    ids = ids[keep]
   
    #print(R['gt'])
    
#    keep = 35
    # sort by confidence
    sorted_ind = np.argsort(-confidence)[:keep]
    sorted_scores = np.sort(-confidence)[:keep]
    BB = BB[sorted_ind, :]
    ids = ids[sorted_ind]

#    print(sorted_scores)
    uni_id = []
    ''' 
    keep = np.where(sorted_scores<-0.1)
    if len(keep[0]) > 0:
        print(keep)
        maj = ids[keep[0]]
        top = min(len(maj),5)
        maj = maj[:top]
        uni_id,uni_count = np.unique(maj, return_counts=True)
        sorted_count = np.argsort(-uni_count)[0]
        uni_id = uni_id[sorted_count]
        print(uni_id)
    print(len(uni_id))
#    print(ids)
    '''           	
    npos = len(list(set(R['gt'])))
    nids = []
    for ii in ids:
        if ii not in nids:
             nids.append(ii)
    ids = nids
    # go down dets and mark TPs and FPs
    nd = len(ids)
    print('No. of detections:{}'.format(nd))
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    area = 0
#    for gt in R['bbox'].astype(float):
#        area += (gt[2] - gt[0] + 1.) * (gt[3] - gt[1] + 1.)
    #print(cachedir+'../data/Images/'+classname+'.jpg')
#    imshape = cv2.imread(cachedir+'/../data/Images/'+classname+'.jpg').shape
    #print(area)
    #print(imshape[0]*imshape[1]*0.4)
#    valid=False
#    if area >= imshape[0]*imshape[1]*0.4:
#        valid=True 
        #print('yes')
#    else:
#        print('lesser boxes')
        
    for d in range(nd):
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        
        inside_gt = False
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            bbarea = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.)
            if any(np.logical_and(inters<bbarea+1. , inters>bbarea-1.)):
                print('inside gt')
                inside_gt=True
      
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
#            print(overlaps)	
#        print(ovmax)
#        print(jmax)
        
        if ids[d] in R['gt']:
            tp[d] = 1
        else:
            fp[d] = 1

        '''            
        if ovmax > ovthresh:
#        if inside_gt :
            if len(uni_id) > 0:
               test_id = uni_id
#               print('Test id:{}'.format(test_id))
            else:
               test_id = ids[d]
            if R['gt'][jmax] == test_id: 
               if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
                
        else:
            fp[d]=1.		
           # if valid and not inside_gt:
           #     fp[d] = 1.
           # else:
           #     fp[d] = 0.
        '''    
    ''' 
    
    uni_id = np.unique(uni_id[0])
    fp = np.zeros(len(uni_id))
    tp = np.zeros(len(uni_id))
    for ind,uid in enumerate(uni_id):
        if uid in R['gt']:
            tp[ind] = 1.
        else:
            fp[ind] = 1.
    tp = np.sort(tp)[::-1]
    fp = np.sort(fp)
    '''
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    
    print(tp)
#    print(tp[-1]/len(tp))
#    print(fp)
    
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = grocery_ap(rec, prec, use_07_metric)
    total.append(ap)
#    print(prec)
#    print(rec)
    ap = np.mean(total)
#    ap = tp[-1]/len(tp)
    return rec, prec, ap
