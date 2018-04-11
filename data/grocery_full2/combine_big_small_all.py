import os
import glob
import sys
import numpy as np


layername = sys.argv[1]
print('Layer:{}'.format(layername))
inters = glob.glob('out_'+layername+'/*.txt')
out_dir = 'combine/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def getMajLabel(overlaps):
    numbers = []
    for label,idx in overlaps:
        numbers.append(int(label))
    counts = {k:numbers.count(k) for k in set(numbers)}
    modes = sorted(dict(filter(lambda x: x[1] == max(counts.values()), counts.items())).keys())
    return modes[-1]



for inter in inters:
    bigbox = []
    with open('results_org_scale_higher_recall/'+os.path.basename(inter)) as ff:
        for line in ff:
            bigbox.append(line.split())
    bigbox = np.array(bigbox)

    outbigbox = []
    with open(inter) as ff:
        for line in ff:
            outbigbox.append(line.split())
    outbigbox = np.array(outbigbox)

    smbox = []
    with open('../../intersection_nmaj_70ov_nms_low_recall/'+os.path.basename(inter)) as ff:
        for line in ff:
            smbox.append(line.split())
    smbox = np.array(smbox)


    smboxmap = {}
    smboxmap[-1] = []
    for bidx in range(bigbox.shape[0]):
        smboxmap[bidx] = []

    with open('out_sift_match_inters_nms_200_300_500c_nmaj_grid_6_patches/'+os.path.basename(inter)) as smf:
        for smidx,line in enumerate(smf):
            idx,label = line.split()
            smboxmap[int(idx)].append((int(label),smidx))
    
    with open(out_dir+os.path.basename(inter),'w') as of:
        for bidx in range(bigbox.shape[0]):
            smoverlaps = smboxmap[bidx]
            if len(smoverlaps) > 0:
                maj_label =  getMajLabel(smoverlaps)
                out_l = 'object_{}'.format(maj_label)
                of.write(out_l+' '+' '.join(bigbox[bidx][1:])+'\n')
#                llll = set([k for k,_ in smoverlaps])
#                for maj_label in llll:
#                    out_l = 'object_{}'.format(maj_label)
#                    of.write(out_l+' '+' '.join(bigbox[bidx][1:])+'\n') 
            else:
                out_l = 'object_{}'.format(outbigbox[bidx][1])
                of.write(out_l+' '+' '.join(bigbox[bidx][1:])+'\n')

        nbbox = smboxmap[-1]
        nlabels = []
        for nbl,nbid in nbbox:
            nlabels.append(int(nbl))

        newbbox = {}
        for k in set(nlabels):
            newbbox[k] = []

        for nbl,nbid in nbbox:
            newbbox[int(nbl)].append(nbid)
        
        for key,val in newbbox.items():
            if len(val) < 2:
                continue
            unionbbox = []
            for ids in val:
                unionbbox.append(smbox[ids][:4])
            unionbbox = np.array(unionbbox).astype(float)
            print(unionbbox.shape)
            xmin = np.min(unionbbox[:,0])
            ymin = np.min(unionbbox[:,1])
            xmax = np.max(unionbbox[:,2])    
            ymax = np.max(unionbbox[:,3])

            out_l = 'object_{}'.format(key)
            of.write(out_l+' 0.7 '+str(xmin)+' ' +str(ymin)+' '+str(xmax)+' ' +str(ymax)+'\n')

#        labelfreq = {k:[nlabels.count(k)] for k in set(nlabels)} 
#        pruned = dict(filter(lambda x: x[1]>1,labelfreq.items()))
#        print(pruned)

         
         
        

               
         
         
