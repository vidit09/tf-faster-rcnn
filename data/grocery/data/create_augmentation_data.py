import os
import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt

out_dir = './data/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

AUG_PER_IMG = 50

product_img_dir = '../freiburg_groceries_dataset/'
product_img_path_file = product_img_dir + 'image_list.txt'

product_files =[]
with open(product_img_path_file) as f:
    product_files = f.readlines()

bg_dir = '../Grocery_products/Testing/store1/'
bg_annotation = bg_dir + 'annotation/'
bg_img = bg_dir + 'images/'

bg_annotation = glob.glob(bg_annotation+'*.txt')
bg_img = glob.glob(bg_img+'*.jpg')

aug_annotation_file = 'annotations.txt'

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

                im_fg = cv2.imread(product_img_dir+product_files[fg_im_id[cnt]].strip())
                row_fg, col_fg, _ = im_fg.shape

                src = np.array([[x, y] for x in [0, col_fg] for y in [0, row_fg]])

                h, status = cv2.findHomography(src, dst)
                im_out = cv2.warpPerspective(im_fg, h, (im_bg.shape[1], im_bg.shape[0]))

                alpha = np.zeros((row, col), dtype='uint8')
                alpha[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])] = 1

                alpha = np.dstack([alpha.T]*3).swapaxes(0, 1).reshape(row, col, 3)

                temp = cv2.multiply(1-alpha, im)
                im = cv2.add(temp, im_out)

                jitter1 = np.random.randint(-10, 10, 1)
                jitter2 = np.random.randint(-10, 10, 1)
                x_min = min(max(int(bbox[0])+jitter1, 0), col-1)
                y_min = min(max(int(bbox[2])+jitter1, 0), row-1)
                x_max = min(max(int(bbox[1])+jitter2, 0), col-1)
                y_max = min(max(int(bbox[3])+jitter2, 0), row-1)

                aug_anno_f.write(str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)+'\n')

                # cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0,255,0),3)
                # plt.imshow(im)

            cv2.imwrite(out_dir+im_name, im)
        # plt.imshow(im)
        # break









