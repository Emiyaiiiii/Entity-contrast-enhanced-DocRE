from info_nce import InfoNCE
import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss, AFLoss
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
from axial_attention.axial_attention import calculate_permutations, PermuteToFrom, SelfAttention
from torch.nn import init

# External Attention《Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks》
class MultiHeadExternalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0 
        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)        
        self.num_heads = self.num_heads * self.coef
        self.k = 128 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)
        

        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x) # B, N, C 
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class SingleHeadExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super(SingleHeadExternalAttention, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
 
    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model
 
        return out

# rewrite axial attention class
class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True, axial_attention = 'self_attention'):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)
        self.axial_attention = axial_attention
        attentions = []
        if axial_attention is 'self_attention':
            for permutation in calculate_permutations(num_dimensions, dim_index):
                attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
        elif axial_attention is 'single_external':
            for permutation in calculate_permutations(num_dimensions, dim_index):
                attentions.append(PermuteToFrom(permutation, SingleHeadExternalAttention(dim)))
        elif axial_attention is 'multi_external':
            for permutation in calculate_permutations(num_dimensions, dim_index):
                attentions.append(PermuteToFrom(permutation, MultiHeadExternalAttention(dim, heads)))          
        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out

class AxialTransformer_by_entity(nn.Module):
    def  __init__(self, emb_size=768, dropout=0.1, num_layers=6, dim_index=-1, heads=8, num_dimensions=2, axial_attention='self_attention'):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_attention = axial_attention
        self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, axial_attention = self.axial_attention) for i in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
    def forward(self, x):
        for idx in range(self.num_layers):
          x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
          x = self.ffns[idx](x)
          x = self.ffn_dropouts[idx](x)
          x = self.lns[idx](x)
        return x

class EntityEnhanceModel(nn.Module):
    def __init__(self, args, config, model, emb_size=768) -> None:
        super().__init__()
        self.config = config
        self.args = args
        self.model = model
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]

        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention
 
    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        b, seq_l, h_size = sequence_output.size()
        n_e = 42
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e: 
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            if len(entity_embs)>0 and len(entity_atts)>0:
                entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
                entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
                s_ne, _ = entity_embs.size()

                ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
                hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
                ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

                pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
                pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
                pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)
                pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)

                h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
                t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
                m = torch.nn.Threshold(0,0)
                ht_att = m((h_att * t_att).mean(1))
                ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
                rs = contract("ld,rl->rd", sequence_output[i], ht_att)

                pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
                pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)

                hss.append(pad_hs) 
                tss.append(pad_ts) 
                rss.append(pad_rs)
        hss = torch.stack(hss, dim=0) 
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        return hss, rss, tss

    def forward(self, input_ids, attention_mask, entity_pos, hts):
        sequence_output, attention = self.encode(input_ids, attention_mask) # sequence_output: [batch\batch*2, max_seq_length, emb], attention: [batch\batch*2, num_layer, max_seq_length, emb]
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=3)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=3)))
        return hs, ts

class EntityPairEnhanceModel(nn.Module):
    def  __init__(self, args, config):
        super().__init__()
        self.config = config
        self.classifier_t = nn.Linear(config.hidden_size , config.num_labels)
        self.axial_transformer = AxialTransformer_by_entity(emb_size = config.hidden_size, dropout=0.0, num_layers=6, heads=8, axial_attention=args.axial_attention) 

    def forward(self, feature, self_mask, nes):
        feature = self.axial_transformer(feature) + feature
        logits_t = self.classifier_t(feature)

        logits_classifier_t = logits_t * self_mask
        logits_classifier_t = torch.cat([logits_classifier_t.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])
        return feature, logits_classifier_t

class DocREModel(nn.Module):
    def __init__(self, args, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.ne = args.ne
        self.enhance = args.enhance

        if self.args.enhance != 'none':
            self.feature_loss_fnt = torch.nn.MSELoss()

            if self.args.evi_loss == 'InfoNCE':
                self.contrast_loss_fnt = InfoNCE()
            elif self.args.evi_loss == 'CosineEmbeddingLoss':
                self.contrast_loss_fnt = torch.nn.CosineEmbeddingLoss()

        if self.args.classifier_loss == 'ATLoss':
            self.loss_fnt = ATLoss()
        elif self.args.classifier_loss == 'AFLoss':
            self.loss_fnt = AFLoss(self.args.gamma_pos, self.args.gamma_neg)

        self.classifier = nn.Linear(config.hidden_size , config.num_labels)  
        self.projection = nn.Linear(emb_size * block_size, config.hidden_size, bias=False) 

        if self.enhance == 'entity_pair':
            self.entityPairEnhanceModel = EntityPairEnhanceModel(args, config)
        if self.enhance == 'context':
            self.entityEnhanceModel = EntityEnhanceModel(args, config, model, emb_size)
        if self.enhance == 'both':
            self.entityPairEnhanceModel = EntityPairEnhanceModel(args, config)
            self.entityEnhanceModel = EntityEnhanceModel(args, config, model, emb_size)

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]

        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention
 
 
    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        b, seq_l, h_size = sequence_output.size()
        n_e = 42
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e: 
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            if len(entity_embs)>0 and len(entity_atts)>0:
                entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
                entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
                s_ne, _ = entity_embs.size()

                ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
                hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
                ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

                pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
                pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
                pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)
                pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)

                h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
                t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
                m = torch.nn.Threshold(0,0)
                ht_att = m((h_att * t_att).mean(1))
                ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
                rs = contract("ld,rl->rd", sequence_output[i], ht_att)

                pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
                pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)

                hss.append(pad_hs) 
                tss.append(pad_ts) 
                rss.append(pad_rs)
        hss = torch.stack(hss, dim=0) 
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                evidence_input_ids=None,
                evidence_attention_mask=None,
                eids_map=None,
                evidence_entity_pos=None,
                evi_hts=None,
                hts_map=None
                ):
                
        sequence_output, attention = self.encode(evidence_input_ids, evidence_attention_mask) # sequence_output: [batch\batch*2, max_seq_length, emb], attention: [batch\batch*2, num_layer, max_seq_length, emb]
        hs, rs, ts = self.get_hrt(sequence_output, attention, evidence_entity_pos, evi_hts)

        # 头实体和尾实体表示
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=3)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=3)))

        b1 = hs.view(-1, self.ne, self.ne, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.ne, self.ne, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(5) * b2.unsqueeze(4)).view(-1, self.ne, self.ne, self.emb_size * self.block_size)

        # 实体对表示
        feature =  self.projection(bl) #[bs, ne, ne, em]
        
        logits_s = self.classifier(feature)

        nes = [len(x) for x in evidence_entity_pos]
        self_mask = (1 - torch.diag(torch.ones((self.ne)))).unsqueeze(0).unsqueeze(-1).to(sequence_output)

        # eval
        logits_classifier_s = logits_s * self_mask
        logits_classifier_s = torch.cat([logits_classifier_s.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])
        output = (self.loss_fnt.get_label(logits_classifier_s, num_labels=self.num_labels),)
        
        if labels is not None: # train
            labels_s = [torch.tensor(label) for label in labels]
            labels_s = torch.cat(labels_s, dim=0).to(logits_classifier_s)
            loss = self.loss_fnt(logits_classifier_s.view(-1, self.config.num_labels).float(), labels_s.float())

            # 上下文表示增强(有效)
            if self.enhance == 'context':
                hs_t, ts_t = self.entityEnhanceModel(input_ids, attention_mask, entity_pos, hts)

                b1_t = hs_t.view(-1, self.ne, self.ne, self.emb_size // self.block_size, self.block_size)
                b2_t = ts_t.view(-1, self.ne, self.ne, self.emb_size // self.block_size, self.block_size)
                bl_t = (b1_t.unsqueeze(5) * b2_t.unsqueeze(4)).view(-1, self.ne, self.ne, self.emb_size * self.block_size)

                feature_t =  self.projection(bl_t) #[bs, ne, ne, em]
                
                logits_t = self.classifier(feature_t)
                logits_classifier_t = logits_t * self_mask
                logits_classifier_t = torch.cat([logits_classifier_t.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])
                
                labels_t = [torch.tensor(label) for label in labels]
                labels_t = torch.cat(labels_t, dim=0).to(logits_classifier_t)
                loss_t = self.loss_fnt(logits_classifier_t.view(-1, self.config.num_labels).float(), labels_t.float())

                target = self.loss_fnt.get_res(self.loss_fnt.get_label(logits_classifier_t, num_labels=self.num_labels), labels_t)
                contrast_loss = self.contrast_loss_fnt(logits_classifier_s, logits_classifier_t, target)

                feature_loss = self.feature_loss_fnt(hs_t, hs) + self.feature_loss_fnt(ts_t, ts)

                loss = loss_t + 0.7 * loss + 0.3 * contrast_loss + 0.3 * feature_loss

            # 实体对表示增强(有效)
            if self.enhance == 'entity_pair':
                feature_t, logits_classifier_t = self.entityPairEnhanceModel(feature, self_mask, nes)
                labels_t = [torch.tensor(label) for label in labels]
                labels_t = torch.cat(labels_t, dim=0).to(logits_classifier_t)
                loss_t = self.loss_fnt(logits_classifier_t.view(-1, self.config.num_labels).float(), labels_t.float())

                target = self.loss_fnt.get_res(self.loss_fnt.get_label(logits_classifier_t, num_labels=self.num_labels), labels_t)
                contrast_loss = self.contrast_loss_fnt(logits_classifier_s, logits_classifier_t, target)
                feature_loss = self.feature_loss_fnt(feature_t, feature)

                loss = loss_t + 0.7 * loss + 0.3 * contrast_loss + 0.3 * feature_loss

            # 实体表示增强 + 实体对表示增强
            if self.enhance == 'both':
                hs_t, ts_t = self.entityEnhanceModel(input_ids, attention_mask, entity_pos, hts)

                b1_t = hs_t.view(-1, self.ne, self.ne, self.emb_size // self.block_size, self.block_size)
                b2_t = ts_t.view(-1, self.ne, self.ne, self.emb_size // self.block_size, self.block_size)
                bl_t = (b1_t.unsqueeze(5) * b2_t.unsqueeze(4)).view(-1, self.ne, self.ne, self.emb_size * self.block_size)

                feature_t =  self.projection(bl_t) #[bs, ne, ne, em]

                # 实体对增强（CosineEmbeddingLoss）
                feature_t, logits_classifier_t = self.entityPairEnhanceModel(feature_t, self_mask, nes)

                labels_t = [torch.tensor(label) for label in labels]
                labels_t = torch.cat(labels_t, dim=0).to(logits_classifier_t)
                loss_t = self.loss_fnt(logits_classifier_t.view(-1, self.config.num_labels).float(), labels_t.float())

                # Divergence loss
                target = self.loss_fnt.get_res(self.loss_fnt.get_label(logits_classifier_t, num_labels=self.num_labels), labels_t)
                contrast_loss = self.contrast_loss_fnt(logits_classifier_s, logits_classifier_t, target)

                # 实体对Features损失
                feature_loss = self.feature_loss_fnt(feature_t, feature) + self.feature_loss_fnt(hs_t, hs) + self.feature_loss_fnt(ts_t, ts)

                loss = loss_t + 0.7 * loss + 0.3 * contrast_loss + 0.3 * feature_loss
            
            output = (loss.to(sequence_output),) + output

        return output