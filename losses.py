import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label #正类标签屏蔽
        n_mask = 1 - labels #负类标签屏蔽

        # Rank positive classes to TH
        # 屏蔽掉除了正类标签以外的标签后对所有正类标签计算损失
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        # 屏蔽掉除了负类标签以外的标签后对所有负类标签计算损失
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

class SimcseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred):
        """有监督的损失函数
        y_pred (tensor): bert的输出, [batch_size * 3, 768]

        """
        # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
        y_true = torch.arange(y_pred.shape[0]).to(y_pred)
        use_row = torch.where((y_true + 1) % 3 != 0)[0]
        y_true = (use_row - use_row % 3 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0]).to(y_pred) * 1e12
        # 选取有效的行
        sim = torch.index_select(sim, 0, use_row)
        # 相似度矩阵除以温度系数
        sim = sim / 0.05
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)
