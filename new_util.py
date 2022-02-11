import numpy as np
import tqdm
import dgl
import os
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn import metrics
path = "data"
np.random.seed(152)

def construct_graph():
    TF_ids = []
    TF_names = []
    Target_ids = []
    Target_names = []
    Disease_ids = []
    Disease_names = []
    f_3 = open(os.path.join(path, "id_TF.txt"), encoding="gbk")
    f_4 = open(os.path.join(path, "id_Target.txt"), encoding="gbk")
    f_5 = open(os.path.join(path, "id_Disease.txt"), encoding="gbk")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        TF_ids.append(identity)
        TF_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break;
        w = w.strip().split()
        identity = int(w[0])
        Target_ids.append(identity)
        Target_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break;
        v = v.strip().split()
        identity = int(v[0])
        paper_name = v[1]
        Disease_ids.append(identity)
        Disease_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    TF_ids_invmap = {x: i for i, x in enumerate(TF_ids)}
    Target_ids_invmap = {x: i for i, x in enumerate(Target_ids)}
    Disease_ids_invmap = {x: i for i, x in enumerate(Disease_ids)}

    TF_Target_src = []
    TF_Target_dst = []
    Target_Disease_src = []
    Target_Disease_dst = []
    TF_Disease_src = []
    TF_Disease_dst = []
    f_1 = open(os.path.join(path, "TF_Target.txt"), "r")
    f_2 = open(os.path.join(path, "Target_Disease.txt"), "r")
    f_0 = open(os.path.join(path, "TF_Disease.txt"), "r")
    # print(len(Target_names),len(TF_names))
    matrix = [([0] * len(Target_names)) for i in range(len(TF_names))]
    for x in f_1:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n'))
        # print(x1,"---",x2)
        matrix[x1][x2] = 1
        
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                TF_Target_src.append(TF_ids_invmap[i])
                TF_Target_dst.append(Target_ids_invmap[j])
    for y in f_2:
        y = y.strip().split()
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        Target_Disease_src.append(Target_ids_invmap[y[0]])
        Target_Disease_dst.append(Disease_ids_invmap[y[1]])
    for ss in f_0:
        ss = ss.strip().split()
        ss[0] = int(ss[0])
        ss[1] = int(ss[1].strip('\n'))
        TF_Disease_src.append(TF_ids_invmap[ss[0]])
        TF_Disease_dst.append(Disease_ids_invmap[ss[1]])
    f_1.close()
    f_2.close()
    f_0.close()

    hg = dgl.heterograph({
        ('TF', 'zt', 'Target') : (TF_Target_src, TF_Target_dst),
        ('Target', 'tz', 'TF') : (TF_Target_dst, TF_Target_src),
        ('Target', 'td', 'Disease') : (Target_Disease_src, Target_Disease_dst),
        ('Disease', 'dt', 'Target') : (Target_Disease_dst, Target_Disease_src),
        ('TF', 'zd', 'Disease') : (TF_Disease_src, TF_Disease_dst),
        ('Disease', 'dz', 'TF') : (TF_Disease_dst, TF_Disease_src)})

    return hg, TF_names, Target_names, Disease_names

def sample(TF_names, Target_names, random_seed):
    matrix = [([0] * len(Target_names)) for i in range(len(TF_names))]
    known_associations = []
    unknown_associations = []
    f_1 = open(os.path.join("data/TF_Target.txt"), "r")
    for x in f_1:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n'))
        matrix[x1][x2] = 1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                known_associations.append([i,j])
            else:
                unknown_associations.append([i,j])

    f_1.close()
    matrix = sp.csr_matrix(matrix)
    npd1 = np.array(unknown_associations)
    npd2 = np.array(known_associations)
    df2 = pd.DataFrame(npd2)
  
    sample_df = df2
    sample_df.reset_index(drop=True, inplace=True)
    unknown_associations = npd1
   
    return sample_df.values, unknown_associations


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Meta_Path_Random_Walk(hg, TF_names, Target_names, Disease_names, num_walks_per_node, walk_length):
    output_path = open(os.path.join(path, "test_output_path.txt"), "w")
    # print(hg)
    '''
    get random walk by 'ztdtz'
    # '''
    for TF_idx in tqdm.trange(hg.number_of_nodes('TF')):
        traces, _ = dgl.sampling.random_walk(
            hg, [TF_idx] * num_walks_per_node, metapath=['zt', 'td', 'dt', 'tz'] * walk_length)
        # 'zt','td','dz','zd','dt', 'tz'
        for tr in traces:
            outline = ""
            for i in range(0, len(tr)):
                if i % 4 == 0:
                    tt = TF_names[tr[i]]
                elif i % 4 == 2:
                    # tt = ""
                    tt = Disease_names[tr[i]]
                else:
                    tt = Target_names[tr[i]]
                outline = outline + ' ' + tt  # skip Disease
            print(outline, file=output_path)
    '''
    get random walk by 'ztdzdtz'
    '''
    # for TF_idx in tqdm.trange(hg.number_of_nodes('TF')):
    #     traces, _ = dgl.sampling.random_walk(
    #         hg, [TF_idx] * num_walks_per_node, metapath=['zt','td','dz','zd','dt', 'tz'] * walk_length)
    #     # 'zt','td','dz','zd','dt', 'tz'
    #     for tr in traces:
    #         outline = ""
    #         for i in range(0, len(tr)):
    #             if i % 3 == 0:
    #                 tt = TF_names[tr[i]]
    #             elif i % 2 == 0:
    #                # tt = ""
    #                 tt = Disease_names[tr[i]]
    #             else:
    #                 tt = Target_names[tr[i]]
    #             outline = outline + ' ' + tt  # skip Disease
    #         print(outline, file=output_path)

    '''
    get random walk by 'tzdzt'
    '''
    # for Target_idx in tqdm.trange(hg.number_of_nodes('Target')):
    #     traces, _ = dgl.sampling.random_walk(
    #         hg, [Target_idx] * num_walks_per_node, metapath=['tz', 'zd', 'dz', 'zt'] * walk_length)
    #     # 'zt','td','dz','zd','dt', 'tz'
    #     for tr in traces:
    #         outline = ""
    #         for i in range(0, len(tr)):
    #             if i % 4 == 0:
    #                 tt = Target_names[tr[i]]
    #             elif i % 4 == 2:
    #                 #tt = ""
    #                 tt = Disease_names[tr[i]]
    #             else:
    #                 tt = TF_names[tr[i]]
    #             outline = outline + ' ' + tt  # skip Disease
    #         print(outline, file=output_path)
    output_path.close()

def getNodeDict():
    TF_ids = []
    TF_names = []
    Target_ids = []
    Target_names = []
    Tf_dict = {}
    Target_dict = {}
    f_3 = open(os.path.join(path, "id_TF.txt"), encoding="gbk")
    f_4 = open(os.path.join(path, "id_Target.txt"), encoding="gbk")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        Tf_dict[z[1]] = identity
        TF_ids.append(identity)
        TF_names.append(z[1])
        # print(Tf_dict)
    while True:
        w = f_4.readline()
        if not w:
            break;
        w = w.strip().split()
        identity = int(w[0])
        Target_dict[w[1]] = identity
        Target_ids.append(identity)
        Target_names.append(w[1])
    # print(Target_dict)
    f_3.close()
    f_4.close()

    return Tf_dict, Target_dict, TF_names, Target_names


def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=True):

    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None,None)
    pred_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            pred_pos.append(sigmoid(score_matrix[edge[0] , edge[1]]))
        else:
            pred_pos.append((score_matrix[edge[0] , edge[1]]))
    # print("pres_pos:" , pred_pos)
    pred_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            pred_neg.append(sigmoid(score_matrix[edge[0],edge[1]]))
        else:
            pred_neg.append(score_matrix[edge[0],edge[1]])
    # calculate scores
    preds_all = np.hstack([pred_pos,pred_neg])
    labels_all = np.hstack([np.ones(len(pred_pos)) , np.zeros(len(pred_neg))])
    roc_score = roc_auc_score(labels_all,preds_all)
    roc_curve_tuple = roc_curve(labels_all,preds_all)

    return roc_score,roc_curve_tuple
