import os
import re

parent_dir = './grocery2/data/'
anno_dir = parent_dir + 'Annotations/'

if not os.path.exists(anno_dir):
    os.mkdir(anno_dir)

with open(parent_dir+'annotations.txt') as f:
    af = None

    for line in f:
        if line.strip() == '#':
            if af is not None:
                af.close()
            im_name = f.readline().strip()
            af = open(anno_dir+im_name[:-4]+'.txt', 'w')

        else:
            coor = re.findall('\d+',line)

            af.write(coor[0]+' '+coor[1]+' '+coor[2]+' '+coor[3]+'\n')


