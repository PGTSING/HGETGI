import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class HeterogeneousNodeEmbedding(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(HeterogeneousNodeEmbedding, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v, emb_size):
        torch.autograd.set_detect_anomaly(True)
        # set the embedding of CS node as all-one vector
        init.constant_(self.v_embeddings.weight[emb_size-1].data, 1)
        init.constant_(self.u_embeddings.weight[emb_size-1].data, 1)

        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)
        
        pos1 = torch.cat((emb_u,emb_v), 1)
        pos2 = torch.cat((emb_v,emb_u), 1)
     
        neg1 = torch.cat((emb_neg_v.squeeze(), emb_u), 1)
        neg2 = torch.cat((emb_u, emb_neg_v.squeeze()), 1)
        
        return pos1, pos2, neg1, neg2

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))