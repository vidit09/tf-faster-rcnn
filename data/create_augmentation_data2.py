import cv2
import numpy as np
import os
import glob
import re

import matplotlib.pyplot as plt

out_dir = 'data2/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_img_dir = out_dir+'Images/'
if not os.path.exists(out_img_dir):
    os.mkdir(out_img_dir)

AUG_PER_IMG = 50

product_img_dir = '/Users/vidit/Documents/Thesis/GroZi-120/'
product_img_path_file = product_img_dir + 'product_file_list.txt'

product_files =[]
with open(product_img_path_file) as f:
    product_files = f.readlines()

bg_dir = '/Users/vidit/Documents/Thesis/Grocery_products/Testing/store1/'
bg_annotation = bg_dir + 'annotation/'
bg_img = bg_dir + 'images/'

bg_annotation = glob.glob(bg_annotation+'*.txt')
bg_img = glob.glob(bg_img+'*.jpg')

aug_annotation_file = out_dir+'annotations.txt'


def create_stacks_from_single_image(im, mask):

    mask = np.dstack([mask.T] * 3).swapaxes(0, 1).reshape(mask.shape[0], mask.shape[1], 3)
    mask[mask == 255] = 1
    cv2.multiply(im, mask, im)

    y = np.where(mask == 1)[0]
    x = np.where(mask == 1)[1]

    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    n_im = im.copy()
    n_mask = mask.copy()

    n_row = np.random.randint(1, 3, 1)[0]
    n_col = np.random.randint(1, 5, 1)[0]

    for r in range(n_col-1):
        n_im = np.concatenate((n_im, im), axis=1)
        n_mask = np.concatenate((n_mask, mask), axis=1)

    m_im = n_im
    m_mask = n_mask

    for c in range(n_row-1):
        n_im = np.concatenate((n_im, m_im), axis=0)
        n_mask = np.concatenate((n_mask, m_mask), axis=0)

    rects = []
    for r in range(n_row):
        for c in range(n_col):
            y_ = im.shape[0]
            x_ = im.shape[1]
            bbox=[]
            bbox.append(x_min+c*x_)
            bbox.append(y_min+r*y_)
            bbox.append(x_max+c*x_)
            bbox.append(y_max+r*y_)

            rects.append(bbox)

    return n_im, n_mask, rects

    # for rect in rects:
    #     cv2.rectangle(n_im, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 3)

    # print(n_im.shape)
    # plt.imshow(n_im)
    # plt.imshow(n_mask*255)
    # plt.show()

def transform_bbox(rects,h):
    rects = np.array(rects)
    rects = rects.reshape((rects.shape[0]*2,2)).T
    ones = np.ones((1,rects.shape[1]))

    rects = np.concatenate((rects,ones))
    transform_rect = h.dot(rects)
    transform_rect[0,:] = transform_rect[0,:]/transform_rect[2,:]
    transform_rect[1,:] = transform_rect[1,:]/transform_rect[2,:]

    transform_rect = transform_rect[0:2,:].T
    transform_rect = transform_rect.reshape(int(transform_rect.shape[0]/2),4)

    return transform_rect


with open(aug_annotation_file,'w') as aug_anno_f:
    for idx, anno_f in enumerate(bg_annotation):
        print('Processing: {}'.format(anno_f))

        with open(anno_f) as anno_file:
            annos = anno_file.readlines()

        num_space = len(annos)
        im_bg = cv2.imread(bg_img[idx])
        row, col, ch = im_bg.shape

        fg_ids = [np.random.choice(len(product_files), len(annos), replace=False) for _ in range(AUG_PER_IMG)]

        for aug_id in range(len(fg_ids)):
            print('augmenting image {}'.format(aug_id))

            fg_im_id = fg_ids[aug_id]
            im = im_bg

            im_name = 'im_{num:04}.jpg'.format(num=idx * len(fg_ids) + aug_id)
            aug_anno_f.write('#\n')
            aug_anno_f.write(im_name+'\n')

            for cnt, anno in enumerate(annos):
                bbox = np.array([float(cor) for cor in anno.strip().split()])
                bbox[0] = bbox[0]*col
                bbox[1] = bbox[1]*col
                bbox[2] = bbox[2]*row
                bbox[3] = bbox[3]*row

                dst = np.array([[int(x), int(y)] for x in bbox[0:2] for y in bbox[2:]])

                im_fg_path = product_img_dir+product_files[fg_im_id[cnt]].strip()
                _im_fg = cv2.imread(im_fg_path)

                _im_name = os.path.basename(im_fg_path)
                im_dir_path = os.path.dirname(im_fg_path)

                im_fg_mask = cv2.imread(im_dir_path+'/../masks/mask'+re.findall('\d+\.png',_im_name)[0],0)

                im_fg, mask_fg, rects = create_stacks_from_single_image(_im_fg, im_fg_mask)

                row_fg, col_fg, _ = im_fg.shape

                src = np.array([[x, y] for x in [0, col_fg] for y in [0, row_fg]])

                h, status = cv2.findHomography(src, dst)
                im_out = cv2.warpPerspective(im_fg, h, (im_bg.shape[1], im_bg.shape[0]))
                mask_fg_out = cv2.warpPerspective(mask_fg, h, (im_bg.shape[1], im_bg.shape[0]))


                n_rects = transform_bbox(rects,h)

                # alpha = np.zeros((row, col), dtype='uint8')
                # alpha[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])] = 1
                #
                # alpha = np.dstack([alpha.T]*3).swapaxes(0, 1).reshape(row, col, 3)

                alpha = mask_fg_out

                temp = cv2.multiply(1-alpha, im)
                im = cv2.add(temp, im_out)

                jitter1 = np.random.randint(-10, 10, 1)
                jitter2 = np.random.randint(-10, 10, 1)

                for bbox in n_rects:
                    x_min = min(max(int(bbox[0])+jitter1, 0), col-1)
                    y_min = min(max(int(bbox[1])+jitter1, 0), row-1)
                    x_max = min(max(int(bbox[2])+jitter2, 0), col-1)
                    y_max = min(max(int(bbox[3])+jitter2, 0), row-1)

                    aug_anno_f.write(str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)+'\n')

                #     cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                # plt.imshow(im)

            cv2.imwrite(out_img_dir+im_name, im)
        # plt.imshow(im)
        # break









