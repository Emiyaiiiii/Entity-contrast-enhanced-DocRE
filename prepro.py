from tqdm import tqdm
import ujson as json
import numpy as np

docred_rel2id = json.load(open('/data/wh/dataset/docred/meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_docred(file_in, tokenizer, max_seq_length=1024 ,axial_attention='none', enhance='none'):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        sents_local = []
        sent_map_local = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            sent_local = []
            new_map_local = {}

            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)

                new_map_local[i_t] = len(sent_local)
                sent_local.extend(tokens_wordpiece)               

            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

            new_map_local[i_t + 1] = len(sents_local)
            sent_map_local.append(new_map_local)
            
            sents_local.append(sent_local)

        train_triple = {}
        evidence_sents = []
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                evidence_sents.extend(label['evidence'])
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})
            for entity in entities:
                evidence_sents.extend([mention['sent_id'] for mention in entity])
            evidence_sents = list(set(evidence_sents))

        entity_pos = [] # all entity_pos in data
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        pos_input_ids = []
        eids_map = [] # a list of entity_id which in evidence sentences
        evidence_entity_pos = [] # a list of entity_pos which in evidence sentences
        if enhance != 'none': # prepare data for evidence sentences contrast training
            if len(evidence_sents)>0:
                for i_s, sent in enumerate(sent_map):
                    sent=list(sent.values())
                    if i_s in evidence_sents:
                        pos_input_ids.extend(tokenizer.convert_tokens_to_ids(sents[sent[0]: sent[-1]]))
            pos_input_ids = tokenizer.build_inputs_with_special_tokens(pos_input_ids)

            for eid, e in enumerate(entities):
                e_poss = []
                for m in e:
                    if m['sent_id'] not in evidence_sents:
                        continue
                    offset = sum([len(sents_local[i]) for i in evidence_sents if i<m['sent_id']]) # local_pos + len(previous sents in evidence)
                    start = sent_map_local[m["sent_id"]][m["pos"][0]] + offset
                    end = sent_map_local[m["sent_id"]][m["pos"][1]] + offset
                    e_poss.append((start, end,))
                if len(e_poss) > 0: # if the entity has at least one mention that occurs in evidence senteces
                    evidence_entity_pos.append(e_poss)
                    eids_map.append(eid)
        
        relations, hts = [], []
        evi_hts, hts_map = [], []
        if axial_attention is 'none': 
            hts_id = 0
            for h, t in train_triple.keys():
                relation = [0] * len(docred_rel2id)
                for mention in train_triple[h, t]:
                    relation[mention["relation"]] = 1
                    evidence = mention["evidence"]
                relations.append(relation)
                hts.append([h, t])
                if h in eids_map and t in eids_map:
                    evi_hts.append([eids_map.index(h), eids_map.index(t)])
                    hts_map.append(hts_id)
                hts_id += 1
                pos_samples += 1
            for h in range(len(entities)):
                for t in range(len(entities)):
                    if h != t and [h, t] not in hts:
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        relations.append(relation)
                        hts.append([h, t])
                        if h in eids_map and t in eids_map:
                            evi_hts.append([eids_map.index(h), eids_map.index(t)])
                            hts_map.append(hts_id)
                        hts_id += 1
                        neg_samples += 1
            assert len(relations) == len(entities) * (len(entities) - 1)
        else: # If we use axial attention, we need to pair up all the entities.
            hts_id = 0
            for h in range(len(entities)):
                for t in range(len(entities)):  
                    if (h, t) in train_triple.keys():
                        relation = [0] * len(docred_rel2id)
                        for mention in train_triple[h, t]:
                            relation[mention["relation"]] = 1
                            evidence = mention["evidence"]
                        relations.append(relation)
                        hts.append([h, t])
                        if h in eids_map and t in eids_map:
                            evi_hts.append([eids_map.index(h), eids_map.index(t)])
                            hts_map.append(hts_id)
                        hts_id += 1
                        pos_samples += 1
                    elif (h, t) not in train_triple.keys():
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        relations.append(relation)
                        hts.append([h, t])
                        if h in eids_map and t in eids_map:
                            evi_hts.append([eids_map.index(h), eids_map.index(t)])
                            hts_map.append(hts_id)
                        hts_id += 1
                        neg_samples += 1
            assert len(relations) == len(entities) * len(entities)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title']
                   }

        if enhance != 'none': # prepare data for evidence sentences contrast training
            feature['pos_input_ids'] = pos_input_ids # context make up by evidence senteces
            feature['eids_map'] = eids_map # entity ids which in evidence senteces
            feature['evidence_entity_pos'] = evidence_entity_pos # entity_pos in evidence senteces
            feature['evi_hts'] = evi_hts # entity pair in evidence senteces 
            feature['hts_map'] = hts_map # entity pair which in evidence senteces  
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features

def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features

def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features

def pseudo_doc2feature(title, evidence, sents_local, entities, sent_map_local, train_triple, tokenizer):
    relations, hts = [], []
    original_hts = []

    pos, neg = 0, 0

    tmp_text = []
    for i in evidence:
        tmp_text.extend(sents_local[i])

    tmp_eids = []
    entity_pos = []
    for eid, e in enumerate(entities):
        e_poss = []
        for m in e:
            if m['sent_id'] not in evidence:
                continue
            offset = sum([len(sents_local[i]) for i in evidence if i<m['sent_id']]) # local_pos + len(previous sents in evidence)
            start = sent_map_local[m["sent_id"]][m["pos"][0]] + offset
            end = sent_map_local[m["sent_id"]][m["pos"][1]] + offset
            e_poss.append((start, end,))

        if len(e_poss) > 0: # if the entity has at least one mention that occurs in evidence
            entity_pos.append(e_poss)
            tmp_eids.append(eid)

    ht2hts_idx = {}
    for new_h, h0 in enumerate(tmp_eids):
        for new_t, t0 in enumerate(tmp_eids):
            if h0 == t0:
                continue

            relation = [0] * len(docred_rel2id)
            if (h0, t0) in train_triple:
                for m in train_triple[h0, t0]:
                    relation[m["relation"]] = 1

            if sum(relation) > 0:
                relations.append(relation)
                ht2hts_idx[(h0,t0)] = len(hts)
                hts.append([new_h, new_t])
                original_hts.append([h0, t0])
                pos += 1
            else:
                relation = [1] + [0] * (len(docred_rel2id) - 1)
                relations.append(relation)
                ht2hts_idx[(h0,t0)] = len(hts)
                hts.append([new_h, new_t])
                original_hts.append([h0, t0])
                neg += 1

    assert( np.all(np.array([len(r) for r in relations]) == 97))
    assert(len(relations) == len(hts))
    # print(len(relations), len(tmp_eids)*(len(tmp_eids) - 1) )
    # assert len(relations) == len(tmp_eids) * (len(tmp_eids) - 1)

    feature = return_feature(tmp_text, entity_pos, relations, hts, title, tokenizer, original_hts=original_hts)

    return feature, pos, neg

def return_feature(tmp_text, entity_pos, relations, hts, title, tokenizer, original_hts=None, max_seq_length=1024):
    tmp_text = tmp_text[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tmp_text) # Returns the vocabulary as a dict of {token: index} pairs
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids) # build model inputs by concatenating and adding special tokens.

    feature = {'input_ids': input_ids, # sents converted by the tokenizer
               'entity_pos': entity_pos, # the [START, END] of each mention of each entity
               'labels': relations, # a list of relations of a pair, each is a one-hot vector
               'hts': hts, # a list of ([h, t]) pairs
               'title': title,
               }

    if original_hts is not None and len(original_hts) > 0:
        feature['original_hts'] = original_hts


    return feature