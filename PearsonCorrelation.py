import torch
import torch.nn as nn
from torch.linalg import matrix_norm as torch_norm

def pearson_correlation_mat(filters):
    flattened_filters = filters.view(filters.shape[0], -1)

    mean = flattened_filters.double().mean(dim=1, keepdim=True)
    std = flattened_filters.double().std(dim=1, keepdim=True)
    normalized_filters = (flattened_filters - mean) / (std + 1e-6)

    return torch.matmul(normalized_filters, normalized_filters.T) / normalized_filters.shape[1]

def mean_filter_correlation_model(model, corrs_l=None):
    with torch.no_grad():
        mean_corr = 0.0
        for i, layer_name in enumerate(model.conv_layers):
            layer = model.get_submodule(layer_name)
            layer_weight = layer.get_weight()

            pearson_mat = pearson_correlation_mat(layer_weight)
            corr_l = (pearson_mat - torch.eye(pearson_mat.shape[0])).abs().max(dim=1).values.mean()
            corr_l = float(corr_l)
            if corrs_l is not None:
                corrs_l[i].append(corr_l)
            mean_corr += corr_l

        return mean_corr / len(model.conv_layers)


def count_mean_filter_correlation_model(mean_corrs=None, corrs_l=None):
    def step(model):
        corr = mean_filter_correlation_model(model, corrs_l)
        if mean_corrs is not None:
            mean_corrs.append(corr)
        return corr

    return step

def correlation_loss(reg_coef, mean_corrs=None, corrs_l=None):
    def loss(_, model):
        # summ = torch.tensor(0.0).to(model.device)
        summ = torch.tensor(0.0)
        mean_corr = 0.0

        for i, layer_name in enumerate(model.conv_layers):
            layer = model.get_submodule(layer_name)
            layer_weight = layer.get_weight()

            pearson_mat = pearson_correlation_mat(layer_weight)
            if corrs_l is not None or mean_corrs is not None:
                corr_l = (pearson_mat - torch.eye(pearson_mat.shape[0])).abs().max(dim=1).values.mean()
                corr_l = float(corr_l)
                if corrs_l is not None:
                    corrs_l[i].append(corr_l)
                if mean_corrs is not None:
                    mean_corr += corr_l

            norm = torch_norm(pearson_mat - torch.eye(pearson_mat.shape[0]))
            summ += torch.square(norm)

        if mean_corrs is not None:
            mean_corrs.append(float(summ / len(model.conv_layers)))

        return reg_coef * summ

    return loss
