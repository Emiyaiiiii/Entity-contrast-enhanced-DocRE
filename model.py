import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss, SimcseLoss
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

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1, axial_attention='none'):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.axial_attention = axial_attention

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        
        if axial_attention is not 'none':
            self.classifier = nn.Linear(config.hidden_size , config.num_labels)
            self.projection = nn.Linear(emb_size * block_size, config.hidden_size, bias=False)
            self.axial_transformer = AxialTransformer_by_entity(emb_size = config.hidden_size, dropout=0.0, num_layers=6, heads=8, axial_attention=self.axial_attention)
        else:
            self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

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
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e: #实体多个提及的起始位置和结束位置
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    #对应原文Encodor通过max pooling处理提及堆叠
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
            #entity_embs是一个实体对内所有涉及实体的实体表示
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            # 头实体是第一列，尾实体是第二列，通过ht_i实体编号获取entity_embs内对应实体的embedding
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            # 同样的方法获取实体的attention
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            # 公式对应论文第四页，通过两个实体的额外本地上下文信息增强实体嵌入
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            # rs为获取到的本地上下文嵌入
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs) # 头实体嵌入
            tss.append(ts) # 尾实体嵌入
            rss.append(rs) # 头实体、尾实体关注的上下文嵌入
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def get_hrt_adjacency_matrix(self, sequence_output, attention, entity_pos, hts):
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
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        

        if self.axial_attention is 'none':
            hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
            #对应修正后的公式3和公式4，即公式6和7
            hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
            ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
            #对应公式5，将hs和ts拆分成多个段，减少参数数量
            b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            logits_classifier = self.bilinear(bl)
        else:
            hs, rs, ts = self.get_hrt_adjacency_matrix(sequence_output, attention, entity_pos, hts)
            ne = 42
            nes = [len(x) for x in entity_pos]

            #对应修正后的公式3和公式4，即公式6和7
            hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=3)))
            ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=3)))
            #对应公式5，将hs和ts拆分成多个段，减少参数数量
            b1 = hs.view(-1, ne, ne, self.emb_size // self.block_size, self.block_size)
            b2 = ts.view(-1, ne, ne, self.emb_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(5) * b2.unsqueeze(4)).view(-1, ne, ne, self.emb_size * self.block_size)

            feature =  self.projection(bl) #[bs, ne, ne, em]
            feature = self.axial_transformer(feature) + feature

            logits = self.classifier(feature)

            self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(0).unsqueeze(-1).to(sequence_output) # mask掉自己和自己的关系对
            logits_classifier = logits * self_mask
            logits_classifier = torch.cat([logits_classifier.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])
        
        output = (self.loss_fnt.get_label(logits_classifier, num_labels=self.num_labels),)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits_classifier)
            loss = self.loss_fnt(logits_classifier.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output

class SimCSEModel(nn.Module):
    def __init__(self, config, pretrained_model, pooling: str):
        super(SimCSEModel, self).__init__()
        self.config = config
        self.model = pretrained_model
        self.pooling = pooling
        self.loss_fnt = SimcseLoss()

    def forward(self, input_ids, attention_mask):
        out, attention, hidden_states = self.encode(input_ids, attention_mask, output_hidden_states=True)
        # out, attention, hidden_states = self.bert(input_ids, attention_mask, output_hidden_states=True)

        if self.pooling == 'cls':
            return self.loss_fnt(hidden_states[-1][:, 0]) # [batch, 768]

        if self.pooling == 'last-avg':
            last = hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            return self.loss_fnt(torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1))  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return self.loss_fnt(torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1))  # [batch, 768]

    def encode(self, input_ids, attention_mask, output_hidden_states=False):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]

        sequence_output, attention, hidden_states = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens, output_hidden_states)
        if output_hidden_states:
            return sequence_output, attention, hidden_states
        else:
            return sequence_output, attention