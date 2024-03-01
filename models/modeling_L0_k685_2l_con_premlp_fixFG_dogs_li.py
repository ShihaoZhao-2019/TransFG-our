# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging 
import math

from os.path import join as pjoin
 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.CELoss import BinaryCrossEntropyLoss

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.top_k = 685
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
 
    def forward(self, hidden_states,forward_counts,layer_num,mask_fb=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
 
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)       # [2,12,1370,1370]
    
        if forward_counts==1 and layer_num==0:
            # print('attention_scores:',attention_scores.shape)
            # print('attention_scores:',attention_scores[0,0,685,630:730])
            # print(layer_num)
            
            attention_scores_01 = attention_scores
            
            vk, _ = torch.topk(attention_scores_01, self.top_k)
            tk = vk[:, :, :, -1].unsqueeze(3).expand_as(attention_scores_01)
            mask_0 = torch.lt(attention_scores_01, tk)
            mask_1 = torch.ge(attention_scores_01, tk)
            attention_scores_01 = attention_scores_01.masked_fill(mask_0, 0).type_as(attention_scores_01)
            attention_scores_01 = attention_scores_01.masked_fill(mask_1, 1).type_as(attention_scores_01)
            
            mask_attn = mask_fb*attention_scores_01
            # mask_attn全局token置1
            mask_attn[:,:,0] = 1
            mask_attn[:,:,:,0] = 1
            
            attention_scores = attention_scores*mask_attn
            # print('post_atten_scores:',attention_scores[0,0,685,630:730])
            # print('attention_scores:',attention_scores)
        else:
            attention_scores = attention_scores
        
        
        attention_probs = self.softmax(attention_scores)
        
        '''
        if forward_counts==1 and layer_num==0:
            # print('attention_probs:',attention_probs.shape)
            print('attention_probs:',attention_probs[0,0,685,630:730])
            # print('mask:',mask[0,0,685,630:730])
            attention_probs = attention_probs+mask
            print('post_atten_scores:',attention_probs[0,0,685,630:730])
            # print('post_atten_scores:',attention_probs.shape)
        else:
            attention_probs = attention_probs
        '''
        
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('position_embeddings',self.position_embeddings)
        # print('x',x)
        # print('position_embeddings',self.position_embeddings.shape)
        # print('x',x.shape)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x,forward_counts,layer_num,mask=None):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x,forward_counts,layer_num,mask)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x, forward_counts, part_weights=None):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        if forward_counts==0:
            last_map = torch.matmul(part_weights, last_map)
        
        weight_map = last_map[:,:,0,:]
        
        last_map = last_map[:,:,0,1:]

        _, max_inx = last_map.max(2)
        return _, max_inx,weight_map

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"] - 1):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention()
        self.part_layer = Block(config)
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states,forward_counts,mask=None):
        attn_weights = []
        for layer,layer_num in zip(self.layer,range(len(self.layer))):
            hidden_states, weights = layer(hidden_states,forward_counts,layer_num,mask)
            attn_weights.append(weights)            
        
        if forward_counts == 0:
            layer_num = -1
            
            part_states, part_weights = self.part_layer(hidden_states,forward_counts,layer_num,mask)
            
            part_num, part_inx, weight_map = self.part_select(attn_weights, forward_counts, part_weights)
            part_inx = part_inx + 1
            # print('part_inx0',part_inx)
        elif forward_counts > 0:
            part_num, part_inx, weight_map = self.part_select(attn_weights, forward_counts)
            part_inx = part_inx + 1
            # print('part_inx1',part_inx)
            parts = []
            B, num = part_inx.shape
            for i in range(B):
                parts.append(hidden_states[i, part_inx[i,:]])
            parts = torch.stack(parts).squeeze(1)
            concat = torch.cat((hidden_states[:,0].unsqueeze(1), parts), dim=1)
            layer_num = -1
            part_states, part_weights = self.part_layer(concat,forward_counts,layer_num,mask)
            
            
        part_encoded = self.part_norm(part_states)   

        return part_encoded, weight_map
    
class mask_Mlp(nn.Module):
    def __init__(self, config,num_tokens):
        super(mask_Mlp, self).__init__()
        self.fc_mask1 = Linear(num_tokens+1, num_tokens+1)
        self.fc_mask2 = Linear(num_tokens+1, num_tokens+1)
        self.mask_fn = ACT2FN["gelu"]
        self.mask_dropout = Dropout(config.transformer["dropout_rate"])

        self.mask_init_weights()

    def mask_init_weights(self):
        nn.init.xavier_uniform_(self.fc_mask1.weight)
        nn.init.xavier_uniform_(self.fc_mask2.weight)
        nn.init.normal_(self.fc_mask1.bias, std=1e-6)
        nn.init.normal_(self.fc_mask2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc_mask1(x)
        x = self.mask_fn(x)
        x = self.mask_dropout(x)
        x = self.fc_mask2(x)
        x = self.mask_dropout(x)
        return x

class wsddn_Mlp(nn.Module):
    def __init__(self, config):
        super(wsddn_Mlp, self).__init__()
        self.fc_wsddn1 = Linear(config.hidden_size, config.hidden_size)
        self.fc_wsddn2 = Linear(config.hidden_size, config.hidden_size)
        self.wsddn_fn = ACT2FN["gelu"]
        self.wsddn_dropout = Dropout(config.transformer["dropout_rate"])

        self.wsddn_init_weights()

    def wsddn_init_weights(self):
        nn.init.xavier_uniform_(self.fc_wsddn1.weight)
        nn.init.xavier_uniform_(self.fc_wsddn2.weight) 
        nn.init.normal_(self.fc_wsddn1.bias, std=1e-6)
        nn.init.normal_(self.fc_wsddn2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc_wsddn1(x)
        x = self.wsddn_fn(x)
        x = self.wsddn_dropout(x)
        x = self.fc_wsddn2(x)
        x = self.wsddn_dropout(x)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids,forward_counts,mask=None):
        embedding_output = self.embeddings(input_ids)
        part_encoded, weight_map = self.encoder(embedding_output,forward_counts,mask)
        return part_encoded, weight_map

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        self.pre_wsddn = wsddn_Mlp(config)
        
        self.hidden_size = config.hidden_size
        # self.part_head = Linear(config.hidden_size, num_classes)
        # self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=config.hidden_size, use_gpu=True)
        self.fc_cls = nn.Linear(in_features=config.hidden_size, out_features=num_classes)
        self.fc_det = nn.Linear(in_features=config.hidden_size, out_features=num_classes)
        
        # self.score_norm_CUB = nn.Conv2d(config.transformer["num_heads"]+1, 1370, kernel_size=1, stride=1, bias=False)
        self.fc_final = nn.Linear(in_features=config.hidden_size, out_features=num_classes)
        
        
        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            self.num_tokens = (448 // patch_size[0]) * (448 // patch_size[1])
        elif config.split == 'overlap':
            self.num_tokens = ((448 - patch_size[0]) // config.slide_step + 1) * ((448 - patch_size[1]) // config.slide_step + 1)
            
        self.mask_heads = config.transformer["num_heads"]
        self.mask_head_size = int(config.hidden_size / config.transformer["num_heads"])
        
        
        '''
        self.pre_wsddn = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = Dropout(config.transformer["dropout_rate"])
        '''
        # self.M_fc = nn.Linear(in_features=self.num_tokens+1, out_features=self.num_tokens+1)
        # self.mask_norm = LayerNorm(self.num_tokens+1, eps=1e-6)
        # self.ffn_mask = mask_Mlp(config,self.num_tokens)
        self.top_k = 685
        print('top_k',self.top_k)
        self.alpha = 1
# =============================================================================
#         self.alpha = 10
# =============================================================================
        print('alpha',self.alpha)
    def forward(self, x, labels=None, one_hot_targets=None):
        forward_counts = 0
        part_tokens, weight_map = self.transformer(x,forward_counts)
        # part_logits = self.part_head(part_tokens[:, 0])
        
        weight_map = weight_map.sum(1).unsqueeze(2)
        # print('weight_map',weight_map.shape)
        weight_map[:,0] =  weight_map[:,1:].max(1)[0]
        
        # 断掉梯度
        weight_map_no_grad = weight_map.detach()
        part_tokens_no_grad = part_tokens.detach()
        
        # wsddn之前
        part_tokens_features = self.pre_wsddn(part_tokens_no_grad)
        # part_tokens_features = part_tokens_no_grad
        
        
        # 传入wsddn
        pooled_features_transfg = part_tokens_features.view(-1,self.hidden_size)
        x_cls = F.softmax(self.fc_cls(pooled_features_transfg), dim=1)                  # [B*1370,200] 
        x_det = self.fc_det(pooled_features_transfg)
        x_det = x_det.view(-1, self.num_tokens+1, self.num_classes)
        x_det = F.softmax(x_det, dim=1)
        x_det = x_det.view(pooled_features_transfg.size(0), -1)                         # [B*1370,200]
        
        # print('x_det',x_det.shape)
        preds_cls = x_cls * x_det                                                       # [B*1370,200]
        preds_cls = preds_cls.view(-1, self.num_tokens+1, self.num_classes)             # [B,1370,200]                                               
        
        preds_cls_weighted = preds_cls*weight_map_no_grad
        
        
        cls_token_weighted = preds_cls_weighted[:,0]
        part_token_weighted = preds_cls_weighted[:,1:]
        
        # print('cls_token_weighted',cls_token_weighted.shape)
        # print('part_token_weighted',part_token_weighted.shape)
        
        
        image_level_scores = part_token_weighted.sum(1)                                           # [B,200]
        
# =============================================================================
#         image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
# =============================================================================
        
        # kl_1 = F.softmax(image_level_scores, dim=1)
        
        # print('image_level_scores',image_level_scores.shape)
        x_cls = x_cls.view(-1, self.num_tokens+1, self.num_classes)       # [B,1370,200]
        x_det = x_det.view(-1, self.num_tokens+1, self.num_classes)
        
        
        # print('x_cls',x_cls)
        # print('x_det',x_det)
        
        # 生成mask
        mask_cls = torch.matmul(x_cls, x_cls.transpose(-1, -2))
        mask_det = torch.matmul(x_det, x_det.transpose(-1, -2))
        mask = mask_cls*mask_det
        # print('mask',mask.shape)
        mask = mask.unsqueeze(1)
        
        
        # 稀疏
        vk, _ = torch.topk(mask, self.top_k)
        # print(value)
        tk = vk[:, :, :, -1].unsqueeze(3).expand_as(mask)
        '''
        mask_01 = torch.lt(mask, tk)
        mask = mask.masked_fill(mask_01, 0).type_as(mask)
        '''
        mask_0 = torch.lt(mask, tk)
        mask_1 = torch.ge(mask, tk)
        mask = mask.masked_fill(mask_0, 0).type_as(mask)
        mask = mask.masked_fill(mask_1, 1).type_as(mask)
        # print('mask0111',mask )
        
        mask_no_grad = mask.detach()
        '''
        mask_grad = mask
        '''
        # print('mask_no_grad',mask_no_grad.shape)
        
        forward_counts = 1
        part_tokens, weight_map = self.transformer(x,forward_counts,mask_no_grad)
        part_logits = self.fc_final(part_tokens[:, 0])
        
        # kl_2 = F.softmax(part_logits, dim=1)
        
        
        # print('part_logits',part_logits.shape)
        
        alpha = self.alpha       # 10e3
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
                loss_cls_token = CrossEntropyLoss()
                loss_wsddn = nn.BCEWithLogitsLoss(reduction='sum') # reduction='sum'
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
             
            cls_token_loss = loss_cls_token(cls_token_weighted.view(-1, self.num_classes), labels.view(-1))
            part_loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
            contrast_loss = con_loss(part_tokens[:, 0], labels.view(-1))
# =============================================================================
#             WSDDN_loss = BinaryCrossEntropyLoss(preds=image_level_scores,
#                                                   targets=one_hot_targets,                              # gt（暂
#                                                   scale_factor=1.0, 
#                                                   size_average=False)
# =============================================================================
            
            '''
            loss_kl = nn.KLDivLoss(reduction='batchmean')
            kl_loss = loss_kl(F.log_softmax(image_level_scores, dim=1),             # dim = 2 使200个类得分归一化为概率
                              F.softmax(part_logits, dim=1))
            '''
            
            
            WSDDN_loss = loss_wsddn(image_level_scores,
                                    one_hot_targets)
            
            # print('part_loss',part_loss)
            # print('WSDDN_loss',WSDDN_loss)
            # print('kl_loss',kl_loss)
            
            # loss = part_loss+WSDDN_loss+cls_token_loss
            loss = part_loss+alpha*(WSDDN_loss+cls_token_loss)+ contrast_loss
            return loss, image_level_scores,part_logits
        else:
            return image_level_scores,part_logits
    
    def transpose_for_mask(self, x):
        new_x_shape = x.size()[:-1] + (self.mask_heads, self.mask_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))


            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
 
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname) 

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
