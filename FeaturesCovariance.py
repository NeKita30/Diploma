import torch
import torch.nn as nn
from torch.linalg import matrix_norm as torch_norm
from torch.linalg import vector_norm as torch_vector_norm

def feat_cov_loss(reg_coef, layers_idx):
    def loss(X, model):
        model.cache_pooled_features(X)
        summ = torch.tensor(0.0)

        for idx in layers_idx:
            data = model.pf_caches[idx][-1]

            mean_feat_of_maps = data.mean(dim=(2,3))
            mean_feat_of_maps_on_batch = mean_feat_of_maps.mean(dim=0)

            deviation_feat_of_maps = mean_feat_of_maps - mean_feat_of_maps_on_batch

            PoCor = (deviation_feat_of_maps.T @ deviation_feat_of_maps) / data.shape[0]

            summ += reg_coef * 1/2 * (torch_norm(PoCor).square()
                                      - torch_vector_norm(torch.diag(PoCor)).square())

        model.drop_pf_caches()
        return summ

    return loss