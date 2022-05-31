# BERT embedding & encoder import
# Vision model이랑 Language model features share architecture

import os

import torch
import torch.utils.data
from torch import nn, Tensor, device
from typing import Tuple

from bert.modeling_bert import BertEmbeddings, BertEncoder, BertPreTrainedModel

from lib.backbone import PatchEmbed, PatchMerging, MMBasicLayer, FusionLayer
from lib.mask_predictor import SimpleDecoding


import torch.nn.functional as F

import torch.utils.checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Model(nn.Module):

    def __init__(self, config,
                pretrain_img_size=512,
                patch_size=4,
                in_chans=3,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                use_checkpoint=False,
                num_heads_fusion=[1, 1, 1, 1],
                fusion_drop=0.0):

        super().__init__()

        self.config = config
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MMBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                #downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                #num_heads_fusion=num_heads_fusion[i_layer],
                #fusion_drop=fusion_drop
            )
            self.layers.append(layer)
        
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = FusionLayer(int(embed_dim * 2 ** i_layer),
                 norm_layer=norm_layer,
                 downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                 num_heads_fusion=num_heads_fusion[i_layer],
                 fusion_drop=fusion_drop)
            layer_name = f'fusion{i_layer}'
            self.add_module(layer_name, layer)
            
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.classifier = SimpleDecoding(8*embed_dim)

        # Text
        self.embeddings = BertEmbeddings(config)
        width = [96, 192, 384, 768] # small
        #width = [128, 256, 512, 1024] # base

        self.encoder = nn.ModuleList([BertEncoder(config, is_cross_attention=True, encoder_width=width[i]) for i in range(4)]) 
        self._init_weights('/database2/ref_seg/pretrained/model_vqa.pth')
        
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
    def _init_weights(self, pretrained=''):
        """ Initialize the weights """
        for name , m in self.named_modules():
            if ('encoder' in name) or ('embeddings' in name): 
                if isinstance(m, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
                    m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif ('layers' in name) or ('fusion' in name) or ('classifier' in name):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                    
        if os.path.isfile(pretrained):
            device = "cuda" if torch.cuda.is_available() else 'cpu'
            pretrained_dict = torch.load(pretrained, map_location = device)['model'] # blip
            model_dict = self.state_dict()
            pretrained_dict_new = {}
            bert_pre = torch.load('/database2/ref_seg/pretrained/pytorch_model.bin',map_location=device)
            #lavt_pre = torch.load('/database2/ref_seg/pretrained/refcoco.pth',map_location=device)['model']
            swin_pre = torch.load('/database2/ref_seg/pretrained/upernet_swin_small_patch4_window7_512x512.pth',map_location=device)['state_dict']
            #print(swin_pre.keys())
            for k, v in swin_pre.items():
                k = k.replace('backbone.','')
                if 'downsample' in k:
                    k = k.replace('layers.','fusion')
                    pretrained_dict_new[k] = v
                elif ('decode' not in k) and ('auxiliary' not in k):
                    pretrained_dict_new[k] = v

            """
            for k, v in lavt_pre.items():
                k = k.replace('backbone.','')
                if 'fusion' in k:
                    k = k.replace('layers.','fusion')
                    pretrained_dict_new[k] = v
                elif 'downsample' in k:
                    k = k.replace('layers.','fusion')
                    pretrained_dict_new[k] = v
                elif 'res_gate' in k:
                    k = k.replace('layers.','fusion')
                    pretrained_dict_new[k] = v
                elif ('norm' in k ) and ('blocks' not in k):
                    pretrained_dict_new[k] = v
            """
            for k, v in bert_pre.items():
                k = k.replace('bert.','')
                if ('embeddings' in k) and ('position_embeddings' not in k):
                    pretrained_dict_new[k] = v
                if 'layer.0' in k:
                    break

            for k, v in pretrained_dict.items():
                if 'visual' in k:
                    continue
                if 'layer.8' in k:
                    break
                
                k = k.replace('text_encoder.','')#replace('bert.','')
                if '0' in k:
                    k = k.replace('layer.0','0.layer.0')
                    if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                        pretrained_dict_new[k] = v
                elif '1' in k:
                    k = k.replace('layer.1','0.layer.1')
                    if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                        pretrained_dict_new[k] = v
                elif '2' in k:
                    k = k.replace('layer.2','1.layer.0')
                    if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                        pretrained_dict_new[k] = v
                elif '3' in k:
                    k = k.replace('layer.3','1.layer.1')
                    if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                        pretrained_dict_new[k] = v
                elif '4' in k:
                    k = k.replace('layer.4','2.layer.0')
                    if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                        pretrained_dict_new[k] = v
                elif '5' in k:
                    k = k.replace('layer.5','2.layer.1')
                    if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                        pretrained_dict_new[k] = v
                elif '6' in k:
                    k = k.replace('layer.6','3.layer.0')
                    #if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                    pretrained_dict_new[k] = v
                elif '7' in k:
                    k = k.replace('layer.7','3.layer.1')
                    #if ('crossattention.self.key' not in k) and ('crossattention.self.value' not in k):
                    pretrained_dict_new[k] = v

            
            model_dict.update(pretrained_dict_new)
            self.load_state_dict(model_dict)
            
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device, is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
   
                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                            causal_mask,
                        ],
                        axis=-1,
                    )                     

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask: Tensor, num_hidden_layers: int, is_attention_chunked: bool = False) -> Tensor:
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to fload if need + fp16 compatibility
        return head_mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """type: torch.Tensor -> torch.Tensor"""
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility

        
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4

        return encoder_extended_attention_mask

    def forward(
        self,
        x, # input image
        input_ids=None, # sentence
        attention_mask=None, # l_mask
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        #encoder_hidden_states=None,
        #encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        v_input_shape = x.shape[-2:]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device, False)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        """
        if encoder_hidden_states is not None:
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [BertPreTrainedModel.invert_attention_mask(mask) for mask in encoder_attention_mask]

            elif encoder_attention_mask is None:
                if type(encoder_hidden_states) == list:
                    encoder_extended_attention_mask = []
                    for hidden in encoder_hidden_states:
                        encoder_batch_size, encoder_sequence_length, _ = hidden.size()
                        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                        if encoder_attention_mask is None:
                            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                            encoder_extended_attention_mask.append(BertPreTrainedModel.invert_attention_mask(encoder_attention_mask))
                else:
                    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                    encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                    encoder_extended_attention_mask = BertPreTrainedModel.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = BertPreTrainedModel.invert_attention_mask(encoder_attention_mask)

        else:
            encoder_extended_attention_mask = None"""

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers, False)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i, layer_module in enumerate(self.encoder):
            v_layer = self.layers[i]
            
            x, H, W = v_layer(x, Wh, Ww)
            image_atts = torch.ones(x.size()[:-1],dtype=torch.long).to(x.device)
            if type(image_atts) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask( mask) for mask in image_atts]
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(image_atts)
            encoder_outputs =  layer_module(embedding_output,
                attention_mask=extended_attention_mask, # l_mask
                head_mask=head_mask,
                encoder_hidden_states=x,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states)
            
            embedding_output = encoder_outputs[0] # l

            if i in self.out_indices:
                fusion_layer = getattr(self,f'fusion{i}')
                
                x_res, H, W, x, Wh, Ww  = fusion_layer(x, H, W, embedding_output.permute(0, 2, 1), attention_mask.unsqueeze(dim=-1))

                norm_layer = getattr(self, f'norm{i}')
                
                x_out = norm_layer(x_res)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out) 
        x_c1, x_c2, x_c3, x_c4 = outs
        x1, x2, x3, x4  = self.classifier(x_c4, x_c3, x_c2, x_c1)

        x_out = F.interpolate(x1, size=v_input_shape, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=v_input_shape, mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=v_input_shape, mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=v_input_shape, mode='bilinear', align_corners=True)
        # encoder outputs : (last layer output, embedding output, layer1 output, layer2 output, layer3 output, ...)

        #sequence_output = encoder_outputs[0]
        #pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        #outputs = (embedding_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return x_out, x2, x3, x4  # sequence_output, (hidden_states), (attentions)