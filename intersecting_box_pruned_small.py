import os
import glob
import numpy as np
from collections import OrderedDict
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

from model.nms_wrapper import nms

do_prune = True
bigbox_results = glob.glob('data/grocery_full2/results_org_scale/*.txt')
smallbox_results = glob.glob('data/grocery_full/results/*.txt')
#smallbox_results = glob.glob('data/grocery5/results_org/*.txt')

out_results = 'intersection/'
if not os.path.exists(out_results):
    os.mkdir(out_results)

def get_maj_label(confs,labels):
    
    if confs.size == 0:
        return -1
    sorted_ind = np.argsort(-confs)
    sorted_labels = labels[sorted_ind]
#    print(sorted_labels)
    uni_label_dict=OrderedDict()
    for l in sorted_labels[:5]:
        if l not in uni_label_dict.keys():
            uni_label_dict[l] = 1
        else:
            uni_label_dict[l] += 1

#    uni_label,uni_count = np.unique(sorted_labels[:5],return_counts=True)
    uni_label = list(uni_label_dict.keys())
    uni_count = list(uni_label_dict.values())
    
    uni_count = [-x for x in uni_count]
    sorted_count_id = np.argsort(uni_count)
    maj_id = uni_label[sorted_count_id[0]]
    return maj_id

bigbox_match_count = 0
smallbox_match_count = 0
num_bigbox = 0
num_smallbox = 0

for i in range(len(bigbox_results)):

    bigbox_path = bigbox_results[i]
    smallbox_path = smallbox_results[i]
    print(bigbox_path)
    print(smallbox_path)
   
    labels=[]
    confs = []
    bboxs=[]
    with open(bigbox_path,'r') as f:
            for line in f:
                     comp = line.split()
                     labels.append(comp[0])
                     confs.append(comp[1])
                     bboxs.append(comp[2:])

    bboxs = np.array(bboxs).astype(float)
    confs = np.array(confs).astype(float)
    labels = np.array(labels)
    interbbox_path = out_results+os.path.basename(bigbox_path)
    
    num_bigbox += len(bboxs)
    match = np.zeros(len(bboxs))

    maj_id = get_maj_label(confs,labels)
    
    smlabels = []
    smconfs = []
    smbboxs = []
    with open(smallbox_path,'r') as f:
        for line in f:
            comp = line.split()
            smlabels.append(comp[0])
            smconfs.append(comp[1])
            smbboxs.append(comp[2:])
    smlabels = np.array(smlabels)
    smbboxs = np.array(smbboxs).astype(np.float32)
    smconfs = np.array(smconfs).astype(np.float32)
   
    if len(smlabels) > 0:
        dets = np.hstack((smbboxs,smconfs[:,np.newaxis])).astype(np.float32,copy=False)
        keep = nms(dets,0.2)
        
        smlabels = smlabels[keep]
        smconfs = smconfs[keep]
        smbboxs = smbboxs[keep,:]


    with open(interbbox_path,'w') as ff:
        for indx in range(len(smlabels)):
            if smconfs[indx]<0.1 and do_prune:
                continue
            bbox = smbboxs[indx,:]
            num_smallbox += 1
            if bboxs.size > 0:

                ixmin = np.maximum(bboxs[:, 0], bbox[0])
                iymin = np.maximum(bboxs[:, 1], bbox[1])
                ixmax = np.minimum(bboxs[:, 2], bbox[2])
                iymax = np.minimum(bboxs[:, 3], bbox[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih


                bbarea = (bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.)
                overlap = inters/bbarea

                id = np.where(overlap>0.7)[0]
                

                if id.size > 0:

                    label = maj_id
                    print(label)

                    smallbox_match_count += 1
                    #id = id[0]
                    for ind in id:
                        
                        ff.write(str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+' ')
                        ff.write(labels[ind]+' '+str(ind)+'\n')
                else:
                   ff.write(str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+' ')
                   ff.write(maj_id+' '+str(-1)+'\n')
