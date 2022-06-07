# BERT embedding & encoder import
# Vision model이랑 Language model features share architecture

import os

import torch
import torch.utils.data
from torch import nn, Tensor, device
from typing import Tuple
from bert.configuration_bert import BertConfig
from bert.modeling_bert import  BertModel, BertEncoder_


from lib.mask_predictor import SimpleDecoding
from lib.backbone import  MultiModalSwinTransformer, MultiModalSwinTransformer_

import torch.nn.functional as F

import torch.utils.checkpoint
from timm.models.layers import  trunc_normal_

class Model2(nn.Module):
    def __init__(self, 
                pretrain_img_size=512,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                num_heads_fusion=[1, 1, 1, 1]):

        super(Model2, self).__init__()
        self.backbone = nn.ModuleList()
        layer = MultiModalSwinTransformer(pretrain_img_size=pretrain_img_size, patch_size=patch_size,  embed_dim=embed_dim, 
                                         depths=depths, num_heads=num_heads,
                                         window_size=window_size, drop_path_rate=0.3, patch_norm=patch_norm,
                                         use_checkpoint=False, num_heads_fusion=num_heads_fusion, i_layer=0
                                         )
        self.backbone.append(layer)
        for i in range(1,4):
            layer = MultiModalSwinTransformer_(embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=window_size,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer,
                 use_checkpoint=use_checkpoint,
                 num_heads_fusion=num_heads_fusion,
                 i_layer=i)
            self.backbone.append(layer)
        self.classifier = SimpleDecoding(8*embed_dim)

        #self.text_encoder = nn.ModuleList()
        config = BertConfig.from_json_file('/database2/ref_seg/pretrained/config.json')
        self.text_encoder1 = BertModel(config)
        #self.text_encoder.append(encoder)

        config1 = BertConfig.from_json_file('/database2/ref_seg/pretrained/config_cross1.json')
        self.text_encoder2 = BertEncoder_(config1, config1.encoder_width)#BertModel_(config)
        #self.text_encoder.append(encoder)

        config2 = BertConfig.from_json_file('/database2/ref_seg/pretrained/config_cross2.json')
        self.text_encoder3 = BertEncoder_(config2, config2.encoder_width)#self.text_encoder3 = BertModel_(config2)
        #self.text_encoder.append(encoder)

        config3 = BertConfig.from_json_file('/database2/ref_seg/pretrained/config_cross3.json')
        self.text_encoder4 = BertEncoder_(config3, config3.encoder_width)#self.text_encoder4 = BertModel_(config3)
        #self.text_encoder.append(encoder)

        self._init_weights()
        
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _init_weights(self):
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        for name , m in self.named_modules():
            if 'backbone' in name:
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

        model_dict = self.state_dict()
        pretrained_dict_new = {}
        #swin_pre = torch.load('/database2/ref_seg/pretrained/swin_base_patch4_window12_384_22k.pth',map_location=device)['model']
        #bert_pre = torch.load('/database2/ref_seg/pretrained/pytorch_model8.bin',map_location=device)
        swin_pre = torch.load('/database2/ref_seg/pretrained/upernet_swin_small_patch4_window7_512x512.pth',map_location=device)['state_dict']
        print("swin~~~")
        for k, v in swin_pre.items():
            #print(k)
            if 'decode_head' in k:
                break
            #k = 'backbone.' + k
            if 'patch_embed' in k:
                k = k.replace('backbone','backbone.0')
                
            if 'layers.0' in k:
                k = k.replace('layers.0','0.layers')
                
            elif 'layers.1' in k:
                k = k.replace('layers.1','1.layers')
                
            elif 'layers.2' in k:
                k = k.replace('layers.2','2.layers')
               
            elif 'layers.3' in k:
                k = k.replace('layers.3','3.layers')
            
            if ('attn_mask' not in k) and ('head' not in k) and ('backbone.norm' not in k):
                pretrained_dict_new[k] = v 
        """print("BERT~~")
        for k, v in bert_pre.items(): 
            if ('embeddings' in k) or ('layer.0' in k) or ('layer.1' in k):
                k = k.replace('bert','text_encoder1')
                pretrained_dict_new[k] = v 
            elif ('layer.2' in k):
                k = k.replace('bert.encoder.layer.2','text_encoder2.layer.0')
                pretrained_dict_new[k] = v 
            elif ('layer.3' in k):
                k = k.replace('bert.encoder.layer.3','text_encoder2.layer.1')
                pretrained_dict_new[k] = v 
            elif ('layer.4' in k):
                k = k.replace('bert.encoder.layer.4','text_encoder3.layer.0')
                pretrained_dict_new[k] = v 
            elif ('layer.5' in k):
                k = k.replace('bert.encoder.layer.5','text_encoder3.layer.1')
                pretrained_dict_new[k] = v 
            elif ('layer.6' in k):
                k = k.replace('bert.encoder.layer.6','text_encoder4.layer.0')
                pretrained_dict_new[k] = v 
            elif ('layer.7' in k):
                k = k.replace('bert.encoder.layer.7','text_encoder4.layer.1')
                pretrained_dict_new[k] = v """

        model_dict.update(pretrained_dict_new)
        self.load_state_dict(model_dict)

    def forward(self, x, l, l_mask):
        input_shape = x.shape[-2:]
        last_hidden_states, att_mask_, head_mask = self.text_encoder1(l, l_mask)
        l_feats = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask_ = l_mask.unsqueeze(dim=-1)
        x1, H, W, x, Wh, Ww = self.backbone[0](x, l_feats, l_mask_)
        image_atts = torch.ones(x.size()[:-1],dtype=torch.long).to(x.device)
        last_hidden_states = self.text_encoder2(last_hidden_states, attention_mask=att_mask_, head_mask=head_mask,
                                                encoder_hidden_states=x, encoder_attention_mask=image_atts)[0]
        #last_hidden_states, att_mask_, head_mask = self.text_encoder2(attention_mask=att_mask_, encoder_hidden_states=x, inputs_embeds=last_hidden_states,
        #                                    encoder_attention_mask=image_atts, head_mask=head_mask)
        ##last_hidden_states = self.text_encoder2(last_hidden_states, attention_mask=att_mask_,  head_mask=head_mask)[0]
        #last_hidden_states, att_mask_, head_mask = self.text_encoder2(inputs_embeds=last_hidden_states, attention_mask=att_mask_,  head_mask=head_mask)
        l_feats = last_hidden_states.permute(0, 2, 1)
        x2, H, W, x, Wh, Ww = self.backbone[1](x, Wh, Ww, l_feats, l_mask_)
        image_atts = torch.ones(x.size()[:-1],dtype=torch.long).to(x.device)
        last_hidden_states = self.text_encoder3(last_hidden_states, attention_mask=att_mask_, head_mask=head_mask,
                                                encoder_hidden_states=x, encoder_attention_mask=image_atts)[0]
        #last_hidden_states, att_mask_, head_mask = self.text_encoder3(attention_mask=att_mask_, encoder_hidden_states=x, inputs_embeds=last_hidden_states,
        #                                    encoder_attention_mask=image_atts,head_mask=head_mask)
        ##last_hidden_states = self.text_encoder3(last_hidden_states, attention_mask=att_mask_,  head_mask=head_mask)[0]
        l_feats = last_hidden_states.permute(0, 2, 1)
        x3, H, W, x, Wh, Ww = self.backbone[2](x, Wh, Ww, l_feats, l_mask_)
        image_atts = torch.ones(x.size()[:-1],dtype=torch.long).to(x.device)
        last_hidden_states = self.text_encoder4(last_hidden_states, attention_mask=att_mask_, head_mask=head_mask,
                                                encoder_hidden_states=x, encoder_attention_mask=image_atts)[0]
        #last_hidden_states, att_mask_, head_mask = self.text_encoder4(attention_mask=att_mask_, encoder_hidden_states=x,inputs_embeds=last_hidden_states,
        #                                    encoder_attention_mask=image_atts, head_mask=head_mask)
        ##last_hidden_states = self.text_encoder4(last_hidden_states, attention_mask=att_mask_,  head_mask=head_mask)[0]
        l_feats = last_hidden_states.permute(0, 2, 1)
        x4, H, W, x, Wh, Ww = self.backbone[3](x, Wh, Ww, l_feats, l_mask_)

        x, x_1, x_2, x_3 = self.classifier(x4,x3,x2,x1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        x_1 = F.interpolate(x_1, size=input_shape, mode='bilinear', align_corners=True)
        x_2 = F.interpolate(x_2, size=input_shape, mode='bilinear', align_corners=True)
        x_3 = F.interpolate(x_3, size=input_shape, mode='bilinear', align_corners=True)

        return x, x_1,x_2, x_3
