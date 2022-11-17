#！/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 22:14
# @Author  :jhin
# @File    :path.py
from collections import defaultdict
import json
import os
import numpy as np
from tqdm import tqdm
# import pandas as pd


def extract_path(data, keep_sent_order):
    # data = data[0]
    sents = data["sents"]

    e2e_nodes = [[] for _ in range(len(data['sents']))]
    mention_nodes = [[] for _ in range(len(data['sents']))]
    #提及所在句子
    m_sent = {}
    # e2e_sent = defaultdict(dict)
    m2m_sent = defaultdict(dict)
    # 实体对应的提及
    e2m = defaultdict(list)
    # 提及对应的实体和句子:提及和实体从1开始，句子从0开始
    m2e_sent = defaultdict(list)
    m2e = {}
    mention_num = -1
    e2sent = defaultdict(list)
    # 文档中存在实体的句子集合
    sent_list = set()
    for i in range(len(data['vertexSet'])):
        e2m[i] = []
        if i not in e2sent:
            e2sent[i] = []
        s_id = [s['sent_id'] for s in data['vertexSet'][i]]
        e2sent[i].append(s_id)
        for s in s_id:
            sent_list.add(s)
        # sent_list.add(data['vertexSet'][i]['sent_id'])
        for j in range(len(data['vertexSet'][i])):

            mention_num += 1
            e2m[i].append(mention_num)
            m_sent[mention_num] =data['vertexSet'][i][j]['sent_id']
            m2e[mention_num] = i
            for key, value in e2m.items():
                for id in range(len(value)):
                    m2e_sent[value[id]] = {}
                    m2e_sent[value[id]][key] = (data['vertexSet'][key][id]['sent_id'])

    # 文档中存在实体的句子的前后组合
    doc_s2s = []
    for i in range(len(sent_list) - 1):
        doc_s2s.append((list(sent_list)[i], list(sent_list)[i + 1]))

    for ns_no, ns in enumerate(data['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            # e2e_nodes[sent_id].append(ns_no+1)
            e2e_nodes[sent_id].append(ns_no)

    for m in m2e_sent:
        for e_key, s_valus in m2e_sent[m].items():
            sent_id = int(s_valus)
            mention_nodes[sent_id].append(m)

    for sent_id in range(len(data["sents"])):
        for n1 in mention_nodes[sent_id]:
            for n2 in mention_nodes[sent_id]:
                if n1 == n2:
                    continue
                if n2 not in m2m_sent[n1]:
                    m2m_sent[n1][n2] = set()
                ####m2m_sent:同一句中不同实体对，即文档中同一句中提及两两组合（待定）,
                m2m_sent[n1][n2].add(sent_id)

    # 具有多个提及的实体所在句组合
    s2s = {}
    # v_list = []
    # 不能通过多跳的句子组合

    no_hop_list = []
    for n1 in e2sent:
        if len(e2sent[n1][0]) <= 1:
            continue

        for s in range(len(e2sent[n1][0]) - 1):

            if abs(list(sent_list).index(e2sent[n1][0][s]) - list(sent_list).index(e2sent[n1][0][s + 1])) != 1:
                continue
            if n1 not in s2s:
                s2s[n1] = []
            s2s[n1].append((e2sent[n1][0][s], e2sent[n1][0][s + 1]))
    v_list = []
    for k, v in s2s.items():
        v_list.append(v[0])
    for i in doc_s2s:
        if i in v_list:
            continue
        no_hop_list.append(i)

    #句内和相邻句间
    one_sentence = defaultdict(dict)
    adjacent_sentence = defaultdict(dict)
    # one_hop = defaultdict(dict)
    for n1 in e2m:
        for n2 in e2m:
            if n1 == n2:
                continue
            for m1 in e2m[n1]:
                for m2 in e2m[n2]:
                    gap = abs(m2e_sent[m1][n1] - m2e_sent[m2][n2])
                    #一句中不同实体的不同提及的组合  ：list(m2e_sent[m2])[0]（实体），m2e_sent[m1][list(m2e_sent[m1])[0]]（句子）
                    if gap == 0:
                        if m2 not in one_sentence[m1]:
                            one_sentence[m1][m2] = []
                            one_sentence[m2][m1] = []
                        one_sentence[m1][m2].append([m_sent[m1]])
                        one_sentence[m2][m1].append([m_sent[m1]])
                    #一跳：不能通过桥实体进行连接到
                    if gap == 1:
                        if m2e_sent[m1][n1] < m2e_sent[m2][n2]:
                            beg, end = m2e_sent[m1][n1], m2e_sent[m2][n2]
                        else:
                            beg, end = m2e_sent[m2][n2], m2e_sent[m1][n1]
                        #只有不连续的两个句子中的提及才能组合，因为其他提及可通过桥提及进行组合
                        if (beg, end) in no_hop_list:
                            if m2 not in adjacent_sentence[m1]:
                                adjacent_sentence[m1][m2] = []
                                adjacent_sentence[m2][m1] = []
                            cand_sents = [beg, end]
                            cand_sents.sort()
                            adjacent_sentence[m1][m2].append([cand_sents])
                            adjacent_sentence[m2][m1].append([cand_sents])

    #一跳：可以通过桥实体进行连接的
    #
    one_hop = defaultdict(dict)


    for n1 in e2m:
        for n2 in e2m:
            if n1 == n2:
                continue
            for m1 in e2m[n1]:
                for m2 in e2m[n2]:
                    for m3 in e2m[n2]:
                        if m2 == m3:
                            continue
                        #1同2 ,3
                        if m2 in one_sentence[m1]:
                            if m2e_sent[m2][n2] == m2e_sent[m3][n2] or m2e_sent[m1][n1] == m2e_sent[m3][n2]:
                                continue
                            if m3 not in one_hop[m1]:
                                one_hop[m1][m3] = []
                                # one_hop[m3][m1] = []
                            cand_sents = [m2e_sent[m1][n1], m2e_sent[m3][n2]]
                            if keep_sent_order == True:
                                cand_sents.sort()
                            one_hop[m1][m3].append((cand_sents, (m2)))
                            # one_hop[m3][m1].append((cand_sents, (m2)))
                        #1邻2 ,3
                        if m2 in adjacent_sentence[m1]:
                            if m2e_sent[m2][n2] == m2e_sent[m3][n2] or m2e_sent[m1][n1] == m2e_sent[m3][n2]:
                                continue
                            if m3 not in one_hop[m1]:
                                one_hop[m1][m3] = []
                                # one_hop_one[m3][m1] = []
                            cand_sents = [m2e_sent[m1][n1], m2e_sent[m2][n2], m2e_sent[m3][n2]]
                            if keep_sent_order == True:
                                cand_sents.sort()
                            one_hop[m1][m3].append((cand_sents, (m2)))
                        # 3 2同1
                        if m1 in one_sentence[m2]:
                            if m2e_sent[m2][n2] == m2e_sent[m3][n2] or m2e_sent[m1][n1] == m2e_sent[m3][n2]:
                                continue
                            if m1 not in one_hop[m3]:
                                one_hop[m3][m1] = []
                                # one_hop[m3][m1] = []
                            cand_sents = [m2e_sent[m1][n1], m2e_sent[m3][n2]]
                            if keep_sent_order == True:
                                cand_sents.sort()
                            one_hop[m3][m1].append((cand_sents, (m2)))
                            # one_hop[m3][m1].append((cand_sents, (m2)))
                        #3 2邻1
                        if m1 in adjacent_sentence[m2]:
                            if m2e_sent[m2][n2] == m2e_sent[m3][n2] or m2e_sent[m1][n1] == m2e_sent[m3][n2]:
                                continue
                            if m1 not in one_hop[m3]:
                                one_hop[m3][m1] = []
                                # one_hop[m3][m1] = []
                            cand_sents = [m2e_sent[m1][n1],m2e_sent[m2][n2], m2e_sent[m3][n2]]
                            if keep_sent_order == True:
                                cand_sents.sort()
                            one_hop[m3][m1].append((cand_sents, (m2)))

                        for m4 in e2m[n1]:
                            if m1 == m4:
                                continue
                            if m2 in one_sentence[m4]:
                                if m2e_sent[m2][n2] == m2e_sent[m3][n2] or m2e_sent[m1][n1] == m2e_sent[m4][n1] or \
                                        m2e_sent[m2][n2] == m2e_sent[m1][n1] or m2e_sent[m1][n1] == m2e_sent[m3][
                                    n2] or m2e_sent[m4][n1] == m2e_sent[m3][n2]:
                                    continue
                                if m3 not in one_hop[m1]:
                                    one_hop[m1][m3] = []
                                    # one_hop[m3][m1] = []
                                cand_sents = [m2e_sent[m1][n1], m2e_sent[m4][n1], m2e_sent[m3][n2]]
                                if keep_sent_order == True:
                                    cand_sents.sort()
                                one_hop[m1][m3].append((cand_sents, (m2)))
                            if m2 in adjacent_sentence[m4]:
                                if m2e_sent[m2][n2] == m2e_sent[m3][n2] or m2e_sent[m1][n1] == m2e_sent[m4][n1] or \
                                        m2e_sent[m2][n2] == m2e_sent[m1][n1] or m2e_sent[m1][n1] == m2e_sent[m3][
                                    n2] or m2e_sent[m4][n1] == m2e_sent[m3][n2]:
                                    continue
                                if m3 not in one_hop[m1]:
                                    one_hop[m1][m3] = []
                                    # one_hop[m3][m1] = []
                                cand_sents = [m2e_sent[m1][n1], m2e_sent[m4][n1], m2e_sent[m2][n2],
                                              m2e_sent[m3][n2]]
                                if keep_sent_order == True:
                                    cand_sents.sort()
                                one_hop[m1][m3].append((cand_sents, (m2)))
                            # one_hop_one[m3][m1].append((cand_sents, (m2)))

    # 两跳：中间的桥提及所在句不能超过头尾提及所在句的最大值
    # mention_path_two = defaultdict(dict)
    two_hop = defaultdict(dict)

    # entityNum = len(data['vertexSet'])
    for n1 in e2m:
        for n2 in e2m:
            if n1 == n2:
                continue
            for n3 in e2m:
                if n3 == n1 or n3 == n2:
                    continue
                for m1 in e2m[n1]:
                    for m2 in e2m[n2]:
                        for m3 in e2m[n3]:
                            ##桥实体只有一个提及
                            if m3 in one_hop[m1] and (m2 in adjacent_sentence[m3] or m2 in one_hop[m3] or m2 in one_sentence[m3]):
                                # for s1 in m2m_sent[m1][m3]:
                                # for s2 in m2m_sent[m3][m2]:
                                if m2e_sent[m1][n1] == m2e_sent[m2][n2]:
                                    continue
                                if m2 not in two_hop[m1]:
                                    two_hop[m1][m2] = []
                                cand_sents = [m2e_sent[m1][n1],m2e_sent[m3][n3], m2e_sent[m2][n2]]
                                if keep_sent_order == True:
                                    cand_sents.sort()
                                two_hop[m1][m2].append((cand_sents, (m3)))
    # 3-hop Path
    three_hop = defaultdict(dict)

    for n1 in e2m:
        for n2 in e2m:
            if n1 == n2:
                continue
            for n3 in e2m:
                if n3 == n1 or n3 == n2:
                    continue
                for n4 in e2m:
                    if n4 == n1 or n4 == n2 or n4 == n3:
                        continue
                    for m1 in e2m[n1]:
                        for m2 in e2m[n2]:
                            for m3 in e2m[n3]:
                                for m4 in e2m[n4]:
                                    # if m3 == m4:
                                    #     continue
                                # 四个提及
                                    if m3 in two_hop[m1] and (m2 in adjacent_sentence[m3] or m2 in one_hop[m3] or m2 in one_sentence[m3]):
                                        if m2e_sent[m1][n1] == m2e_sent[m2][n2]:
                                            continue
                                        if m2 not in three_hop[m1]:
                                            three_hop[m1][m2] = []
                                        cand_sents = [m2e_sent[m1][n1], m2e_sent[m3][n3], m2e_sent[m2][n2]]
                                        if keep_sent_order == True:
                                            cand_sents.sort()
                                        three_hop[m1][m2].append((cand_sents, (m3)))
                                    if (m3 in adjacent_sentence[m1] or m3 in one_hop[m1] or m3 in one_sentence[m1]) and m2 in two_hop[m3]:
                                        if m2e_sent[m1][n1] == m2e_sent[m2][n2]:
                                            continue
                                        if m2 not in three_hop[m1]:
                                            three_hop[m1][m2] = []
                                        cand_sents = [m2e_sent[m1][n1], m2e_sent[m3][n3], m2e_sent[m2][n2]]
                                        if keep_sent_order == True:
                                            cand_sents.sort()
                                        three_hop[m1][m2].append((cand_sents, (m3)))

                                    if (m3 in adjacent_sentence[m1] or m3 in one_hop[m1] or m3 in one_sentence[m1]) and (m4 in adjacent_sentence[m3] or m4 in one_hop[m3] or m4 in one_sentence[m3]) and (m2 in adjacent_sentence[m4] or m2 in one_hop[m4] or m2 in one_sentence[m4]) :
                                        if m2e_sent[m1][n1] == m2e_sent[m2][n2]:
                                            continue
                                        if m2 not in three_hop[m1]:
                                            three_hop[m1][m2] = []
                                        cand_sents = [m2e_sent[m1][n1], m2e_sent[m3][n3], m2e_sent[m4][n4], m2e_sent[m2][n2]]
                                        if keep_sent_order == True:
                                            cand_sents.sort()
                                        three_hop[m1][m2].append((cand_sents, (m3)))

    other = defaultdict(dict)
    # Merge
    merge = defaultdict(dict)
    for n1 in e2m:
        for n2 in e2m:
            if n1 == n2:
                continue
            for m1 in e2m[n1]:
                for m2 in e2m[n2]:
                    if m2 in one_sentence[m1]:
                        if m2 in merge[m1]:
                            merge[m1][m2] += one_sentence[m1][m2]
                        else:
                            merge[m1][m2] = one_sentence[m1][m2]
                    if m2 in adjacent_sentence[m1]:
                        if m2 in merge[m1]:
                            merge[m1][m2] += adjacent_sentence[m1][m2]
                        else:
                            merge[m1][m2] = adjacent_sentence[m1][m2]
                    if m2 in two_hop[m1]:
                        if m2 in merge[m1]:
                            merge[m1][m2] += two_hop[m1][m2]
                        else:
                            merge[m1][m2] = two_hop[m1][m2]

                    if m2 in three_hop[m1]:
                        if m2 in merge[m1]:
                            merge[m1][m2] += three_hop[m1][m2]
                        else:
                            merge[m1][m2] = three_hop[m1][m2]
                    #
                    if m2 in one_hop[m1]:
                        if m2 in merge[m1]:
                            merge[m1][m2] += one_hop[m1][m2]
                        else:
                            merge[m1][m2] = one_hop[m1][m2]
                    #
                    else:
                        sents = set()
                        mmlist = []
                        if m2 not in other[m1]:
                            other[m1][m2] = []
                        for i in data['vertexSet'][n1]:
                            for j in data['vertexSet'][n2]:
                                mmlist.append([i['sent_id'],j['sent_id']])
                        for i in mmlist:
                            for j in i:
                                sents.add(j)
                        cand_sents = [i for i in sents]
                        if keep_sent_order == True:
                            cand_sents.sort()
                            other[m1][m2].append((cand_sents, (m1, m2)))
                        if m2 in other[m1]:
                            if m2 in merge[m1]:
                                merge[m1][m2] += other[m1][m2]
                            else:
                                merge[m1][m2] = other[m1][m2]
    evi_merge = defaultdict(dict)
    # 将找到的目标提及对转换为实体对
    e_merge = set()
    for h in merge:
        for t in merge[h]:
            e_merge.add((list(m2e_sent[h])[0], list(m2e_sent[t])[0]))
            if list(m2e_sent[t])[0] not in evi_merge[list(m2e_sent[h])[0]]:
                evi_merge[list(m2e_sent[h])[0]][list(m2e_sent[t])[0]] = []
            evi_merge[list(m2e_sent[h])[0]][list(m2e_sent[t])[0]].append(merge[h][t])
    e_merge = list(e_merge)
    e_merge.sort()
    # 对证据句进行整合
    evi = defaultdict(dict)
    for i in evi_merge:
        for j in evi_merge[i]:
            n = set()
            for k in range(len(evi_merge[i][j])):
                for p in range(len(evi_merge[i][j][k])):
                    if isinstance(evi_merge[i][j][k][0][0], int):
                        n.add(evi_merge[i][j][k][0][0])
                    if isinstance(evi_merge[i][j][k][0][0], list):
                        for l in evi_merge[i][j][k][0][0]:
                            n.add(l)
            if j not in evi[i]:
                evi[i][j] = []
            evi[i][j].append(list(n))
    # 具有相同证据句的实体对
    ht_evi = defaultdict(dict)
    for h in evi:
        for t in evi[h]:
            evi_str = ""
            evis = list(evi[h][t][0])
            for e_i in evis:
                evi_str = evi_str + str(e_i) + ","
            if evi_str not in ht_evi:
                ht_evi[evi_str] = []
            # ht_evi[evi_str].append((h, t))
            ht_evi[evi_str].append([h, t])
    return ht_evi

