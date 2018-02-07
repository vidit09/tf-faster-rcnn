#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'object')
#CLASSES_FULL = tuple(['__background__'])+tuple(['object_'+str(i+1) for i in range(27)])
CLASSES_FULL = ('__background__',
                 'Bakery','Biscuits','Candy/Bonbons',
'Candy/Chocolate','Cereals','Chips',
'Coffee','Dairy/Cheese','Dairy/Creme',
'Dairy/Yoghurt','DriedFruitsAndNuts',
'Drinks/Choco','Drinks/IceTea',
'Drinks/Juices','Drinks/Milk',
'Drinks/SoftDrinks','Drinks/Water',
'Jars-Cans/Canned','Jars-Cans/Sauces',
'Jars-Cans/Spices','Jars-Cans/Spreads',
'Oil-Vinegar','Pasta','Rice',
'Snacks','Soups','Tea')



NETS = {'res101': ('res101_faster_rcnn_iter_{}.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
           'grocery':('grocery_train',),'grocery2':('grocery2_train',),'grocery3':('grocery3_train',), 'grocery4':('grocery4_train',),
'grocery5':('grocery5_train',), 'grocery6':('grocery6_train',),'grocery7':('grocery7_train',),'grocery8':('grocery8_train',),
           'grocery9':('grocery9_train',),'grocery10':('grocery10_train',),'grocery_full':('grocery_full_train',),
           'grocery_full2':('grocery_full2_train',),'grocery_full3':('grocery_full3_train',)}

def vis_detections(im, class_name, dets,im_path, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        #print('Total Proposed bbox:{}'.format(dets.shape[0]))
        #print('Total Proposed bbox after threshold:{}'.format(len(inds)))
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # fig.set_visible(False)
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print("Detection Probability:{}".format(score))
        print("Class:{}".format(class_name))
        sub_mat = im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),2].copy().astype(int)
        sub_mat += 200
        sub_mat = (sub_mat*255/np.max(sub_mat)).astype(np.uint)
        im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), 1] = sub_mat
        # bbox_pts = np.array([[x, y] for x in [bbox[0], bbox[2]] for y in [bbox[1], bbox[3]]])
        # cv2.fillPoly(im,bbox_pts,(0,255,0))
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,0),3)
        cv2.putText(im,'{}'.format(class_name),(int(bbox[0]),int(bbox[1]+50)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    #     ax.add_patch(
    #         plt.Rectangle((bbox[0], bbox[1]),
    #                       bbox[2] - bbox[0],
    #                       bbox[3] - bbox[1], fill=False,
    #                       edgecolor='red', linewidth=3.5)
    #         )
    #     ax.text(bbox[0], bbox[1] - 2,
    #             '{:s} {:.3f}'.format(class_name, score),
    #             bbox=dict(facecolor='blue', alpha=0.5),
    #             fontsize=14, color='white')
    #
    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()

    path = os.path.dirname(im_path)
    im_name = os.path.basename(im_path)
    out_image = path+'/'+im_name.split('.')[0]+'_out.jpg'
    print('Saving output at {}'.format(out_image))
    cv2.imwrite(out_image,im)

def demo(sess, net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    print(im.shape)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    print(scores.shape)
    print(len(classes))
    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, im_file, thresh=CONF_THRESH)
#        break
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    parser.add_argument('--iters', dest='iters')
    parser.add_argument('--img', dest='img')
    parser.add_argument('--full',dest='full',default='0')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    iters = args.iters
    img = args.img
    if args.full == '0':
        classes = CLASSES
    else:
        classes = CLASSES_FULL
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0].format(iters))


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", len(classes),
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    train_writer = tf.summary.FileWriter('./', tf.get_default_graph())
    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['7.jpg']
    im_names = [img]
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name, classes)

    # plt.show()
