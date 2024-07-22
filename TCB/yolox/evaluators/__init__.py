#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco_evaluator import COCOEvaluator


from .mot_evaluator_dancetrack import MOTEvaluator
#from .mot_evaluator_mot17 import MOTEvaluator
#from .mot_evaluator_mot20 import MOTEvaluator



#from .mot_evaluator import MOTEvaluator
#from .mot_evaluator_dance import MOTEvaluator
#from .mot_evaluator_ocsort_mot import MOTEvaluator
from .mot_evaluator_bdd import BDDEvaluator
from .evaluation import Evaluator
