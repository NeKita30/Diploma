import torch
import torch.nn as nn

def projection(a, b):
    return (torch.dot(a.view(-1), b.view(-1))
            / torch.dot(b.view(-1), b.view(-1))) * b

def gram_schmidt(f_a, proj, lr, wort):
    return f_a - lr * wort * proj


def ortho_step(wort, optim, max_projections=None):
    def step(model):
        lr = optim.param_groups[0]['lr']

        with torch.no_grad():
            for l_i, name in enumerate(model.conv_layers):
                layer = model.get_submodule(f"{name}")
                layer_weight = layer.get_weight()
                k = len(layer_weight)
                a, b = 0, 1  # the nearest filters indexes
                max_projection = projection(layer_weight[a], layer_weight[b])
                for i in range(k):
                    for j in range (i + 1, k):
                        proj = projection(layer_weight[i], layer_weight[j])

                        if torch.norm(max_projection) < torch.norm(proj):
                            a, b = i, j
                            max_projection = proj
                if max_projections is not None:
                    max_projections[l_i].append(torch.norm(max_projection))

                layer_weight[a] = gram_schmidt(layer_weight[a],
                                               max_projection, lr, wort)

    return step


def pseudo_ortho(wort, optimizer, max_projections):
    def step(model):
        with torch.no_grad():
            for l_i, name in enumerate(model.conv_layers):
                layer = model.get_submodule(f"{name}")
                layer_weight = layer.get_weight()
                k = len(layer_weight)
                a, b = 0, 1  # the nearest filters indexes
                max_projection = projection(layer_weight[a], layer_weight[b])
                for i in range(k):
                    for j in range (i + 1, k):
                        proj = projection(layer_weight[i], layer_weight[j])

                        if torch.norm(max_projection) < torch.norm(proj):
                            a, b = i, j
                            max_projection = proj

                max_projections[l_i].append(torch.norm(max_projection))
    return step
