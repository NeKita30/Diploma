import datetime
import json
import os

import torch

from classification_utils import test, Statistics

MODELS_PATH = "./models"
STATISTICS_PATH = "./statistics"
RESULTS_PATH = "./results"

for pt in (MODELS_PATH, STATISTICS_PATH, RESULTS_PATH):
    if not os.path.exists(pt):
        os.mkdir(pt)


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'shape': obj.shape, 'data': obj.detach().cpu().view([-1]).numpy().tolist()}
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_model(model, name, test_loader, train_time=None, device='cpu'):
    model = model.to(device)
    _, accuracy1 = test(model, test_loader, device=device, k_max=1)
    _, accuracy5 = test(model, test_loader, device=device, k_max=5)
    torch.save(model.cpu().state_dict(), f'{MODELS_PATH}/{name}.pt')
    with open(f'{MODELS_PATH}/{name}.json', 'w') as json_file:
        json.dump(model.state_dict(), json_file, cls=TensorEncoder, indent=2)
    with open(f'{RESULTS_PATH}/{name}.json', 'w') as json_file:
        result = {
            'top1': accuracy1,
            'top5': accuracy5,
        }
        if train_time is not None:
            result['train_time'] = str(datetime.timedelta(seconds=train_time))
        json.dump(result, json_file, cls=TensorEncoder, indent=2)


def save_statistics(name, train_stat: Statistics, test_stat: Statistics):
    with open(f'{STATISTICS_PATH}/{name}.json', 'w') as json_file:
        results = {}
        if train_stat is not None:
            results['train_idx'] = train_stat.idx
            results['train_loss'] = train_stat.loss
            results['train_accuracy'] = train_stat.accuracy
        if test_stat is not None:
            results['test_idx'] = test_stat.idx
            results['test_loss'] = test_stat.loss
            results['test_accuracy'] = test_stat.accuracy
        json.dump(results, json_file, indent=2)