from new_util import construct_graph, sample, get_roc_score, Meta_Path_Random_Walk, getNodeDict
import numpy as np
from sklearn import metrics
import tqdm
import dgl
import torch
from Metapath import EmbeddingTrainer
from sklearn.model_selection import KFold
num_walks_per_node = 350    
walk_length = 130
path = "data"

def Train(path, output_file, dim, window_size, iterations, batch_size, care_type, initial_lr, min_count, num_workers,
          random_seed):
    hg, TF_names, Target_names, Disease_names = construct_graph()
    print(hg.edges('all', etype = 'zt'))
    print(hg.edges('all', etype = 'tz'))
    print(hg.edges('all', etype = 'td'))
    print(hg.edges('all', etype = 'dt'))
    print(hg.edges('all', etype = 'zd'))
    print(hg.edges('all', etype = 'dz'))

    _ , _ , eid = hg.edges('all', etype = 'zt')
    # print(eid)
    # print(len(eid))
    samples, unknowassociation = sample(TF_names, Target_names, random_seed=1)
    random_seed=1
    kf = KFold(n_splits = 5 , shuffle = True)
    train_index = []
    test_index = []

    for train_idx, test_idx in kf.split(samples):
        print("--------------------------------------------------------------------------")
        print(len(train_idx),"train_index:",train_idx)
        print(len(test_idx),"test_index:",test_idx)
        train_index.append(train_idx)
        test_index.append(test_idx)

    print(len(train_index[0]))
    print(len(test_index[0]))
    test_edges_false = []
    for xx in unknowassociation:
        test_edges_false.append(xx)
    # remove_index = []

    auc_result = []
    fprs = []
    tprs = []
    
    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)
        remove_index = []
        hg, TF_names, Target_names, Disease_names = construct_graph()

        for j in test_index[i]:
            if j >= len(eid):
                continue
            remove_index.append(j)
        remove_index = torch.tensor(remove_index, dtype=torch.int64)

        hg.remove_edges(remove_index, 'zt')
        hg.remove_edges(remove_index, 'tz')
    
        train_edge_false = []
        train_edge = []
        for xx in train_index[i]:
            if xx >= len(eid):
                train_edge_false.append(samples[xx])
            else:
                train_edge.append(samples[xx])
        # print("train_edge_false:",train_edge_false)
        # print("train_edge:",train_edge)

        # test_edges_false = []
        test_edge = []
        for xx in test_index[i]:
            test_edge.append(samples[xx])

        # print("test_edges_false:",test_edges_false)
        # print("test_edge:",test_edge)
        # print(len(test_edge))
        # print(len(test_edges_false))

        Meta_Path_Random_Walk(hg, TF_names, Target_names, Disease_names, num_walks_per_node, walk_length)

        m2v = EmbeddingTrainer(path=path, output_file=output_file, dim=dim, window_size=window_size, iterations=iterations,
                                  batch_size=batch_size, care_type=care_type, initial_lr=initial_lr, min_count=min_count, num_workers=num_workers)
        m2v.train()

        Tf_dict, Target_dict, TF_names, Target_names = getNodeDict()
        file1 = open("TF_Target_Disease/output_first")
        TF_embed_dict = {}
        Target_embed_dict = {}
        file1.readline()
        for line in file1:
            embed = line.strip().split(' ')
            if embed[0] in TF_names:
                TF_embed_dict[embed[0]] = []
                for i in range(1,len(embed),1):
                    TF_embed_dict[embed[0]].append(float(embed[i]))
            if embed[0] in Target_names:
                Target_embed_dict[embed[0]] = []
                for i in range(1,len(embed),1):
                    Target_embed_dict[embed[0]].append(float(embed[i]))
        TF_emb_list = []
        for node in Tf_dict.keys():
            node_emb = TF_embed_dict[node]
            TF_emb_list.append(node_emb)
        TF_emb_matrix = np.vstack(TF_emb_list)
        # print(TF_emb_matrix.shape)

        Target_emb_list = []
        for node1 in Target_dict.keys():
            node_emb1 = Target_embed_dict[node1]
            Target_emb_list.append(node_emb1)
        Target_emb_matrix = np.vstack(Target_emb_list)
        score_matrix = np.dot(TF_emb_matrix,Target_emb_matrix.T)
        test_roc, test_roc_curve = get_roc_score(test_edge,test_edges_false,score_matrix,apply_sigmoid=True)

        fpr = test_roc_curve[0]
        tpr = test_roc_curve[1]
        # np.savetxt('score_matrix.txt', score_matrix, fmt='%s', delimiter=',')
        test_auc = metrics.auc(fpr,tpr)

        auc_result.append(test_auc)
        fprs.append(fpr)
        tprs.append(tpr)

    print("## Training Finished!")
    print("--------------------------------------------------------------------------")
    return auc_result, fprs, tprs
