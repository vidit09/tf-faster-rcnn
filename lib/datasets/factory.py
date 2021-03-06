# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.grocery import grocery
from datasets.grocery_full import grocery_full
import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))


# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery'

for split in ['train', 'val', 'test']:
  name = 'grocery_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery2'

for split in ['train', 'val', 'test']:
  name = 'grocery2_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='2'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery3'

for split in ['train', 'val', 'test']:
  name = 'grocery3_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='3'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery4'

for split in ['train', 'val', 'test']:
  name = 'grocery4_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='4'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery5'

for split in ['train', 'val', 'test']:
  name = 'grocery5_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='5'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery6'

for split in ['train', 'val', 'test']:
  name = 'grocery6_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='6'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery7'

for split in ['train', 'val', 'test']:
  name = 'grocery7_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='7'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery_full'

for split in ['train', 'val', 'test']:
  name = 'grocery_full_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery_full(split, path))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery8'

for split in ['train', 'val', 'test']:
  name = 'grocery8_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='8'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery9'

for split in ['train', 'val', 'test']:
  name = 'grocery9_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='9'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery10'

for split in ['train', 'val', 'test']:
  name = 'grocery10_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='10'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery11'

for split in ['train', 'val', 'test']:
  name = 'grocery11_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery(split, path, db='11'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery_full2'

for split in ['train', 'val', 'test']:
  name = 'grocery_full2_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery_full(split, path, db='2'))

# Set own dataset
datapath = os.path.dirname(os.path.abspath(__file__))+'/../../data/grocery_full3'

for split in ['train', 'val', 'test']:
  name = 'grocery_full3_{}'.format(split)
  __sets[name] = (lambda split=split,path=datapath: grocery_full(split, path, db='3'))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
