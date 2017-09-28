import os
import re

anno_dir = 'Annotations/'

if not os.path.exists(anno_dir):
    os.mkdir(anno_dir)

with open('./annotations.txt') as f:
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


