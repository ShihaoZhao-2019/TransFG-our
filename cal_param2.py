# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:25:29 2021

@author: tyy
"""

from torchstat import stat
import torchvision.models as models
from thop import profile
from ptflops import get_model_complexity_info
import torch 
from models.modeling_TransFG_fixFG import CONFIGS

# =============================================================================
# import models.modeling_TransFG_fixFG.VisionTransformer as VisionTransformer_FG
# import models.modeling_L0_k685_conloss_premlp.VisionTransformer as VisionTransformer_our
# =============================================================================
from models.modeling_TransFG_fixFG import VisionTransformer as VisionTransformer_FG
from models.modeling_OUR_FLOPS import VisionTransformer as VisionTransformer_our
from models.modeling_ViT_topk import VisionTransformer as VisionTransformer_topk
from models.modeling_ViTcover_fixFG_li import VisionTransformer as VisionTransformer_VIT
import models.models_att as models_att
# Prepare model
config = CONFIGS['ViT-B_16']
config.split = 'overlap'
config.slide_step = 12


num_classes = 200


model_FG = VisionTransformer_FG(config, 448, zero_head=True, num_classes=num_classes, smoothing_value=0.0)
model_VIT = VisionTransformer_FG(config, 448, zero_head=True, num_classes=num_classes, smoothing_value=0.0)
model_our = VisionTransformer_our(config, 448, zero_head=True, num_classes=num_classes, smoothing_value=0.0)
model_topk = VisionTransformer_topk(config, 448, zero_head=True, num_classes=num_classes, smoothing_value=0.0)
model_att = models_att.buildCNN()

macs, params = get_model_complexity_info(model_FG, (3, 448, 448), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print('FG  ','flops:',macs,'\n','params:',params)
print('************************************************')
macs, params = get_model_complexity_info(model_VIT, (3, 448, 448), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print('VIT  ','flops:',macs,'\n','params:',params)
print('************************************************')
macs, params = get_model_complexity_info(model_our, (3, 448, 448), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print('our  ','flops:',macs,'\n','params:',params)

print('************************************************')
macs, params = get_model_complexity_info(model_topk, (3, 448, 448), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print('topk  ','flops:',macs,'\n','params:',params)
print('************************************************')

macs, params = get_model_complexity_info(model_att, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print('att  ','flops:',macs,'\n','params:',params)
print('************************************************')
