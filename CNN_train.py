from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import time

from classification_utils import *
from cifar_models import CNN6, CNN8
from quantized_layers import *
from model_saving import *

from Orthogonalization import *
from PearsonCorrelation import *
from FeaturesCovariance import *

MODEL_ARCH = CNN8

def ptq_model(model, in_bins: int, w_bins: int, symmetric_mode: bool,
              train_loader, device='cpu', mode='ada'):
    model = model.to(device)
    qmodel = MODEL_ARCH()
    qmodel.load_state_dict(model.state_dict())
    qmodel = qmodel.to(device)

    cmodel = MODEL_ARCH()
    cmodel.load_state_dict(model.state_dict())
    cmodel = cmodel.to(device)
    N_BLOCKS = 50

    cache = []
    for i, (X, _) in enumerate(train_loader):
        if i == N_BLOCKS:
            break
        cache.append(X.to(device))
        cmodel.cache(X.to(device))

    in_step = -1
    for i, layer_name in enumerate(qmodel.layer_iterator()):
        layer = getattr(cmodel, layer_name)
        corrected_in = None
        if i > 0:
            for X in cache:
                qmodel.cache(X)
            prev_layer = getattr(qmodel, layer_name)
            corrected_in = prev_layer.cached_in
        qlayer = QConv2d(layer, in_bins, w_bins, symmetric_mode, in_step, mode, corrected_in).to(device)
        in_step = qlayer.input_quant.scale * qlayer.weight_quant.scale
        setattr(qmodel, layer_name, qlayer)
        qmodel.drop_cache()
    del cmodel
    return qmodel


def gradual_freeze_model(qmodel, train_loader, test_loader, device, model_update):
    EP_FREEZE = 3
    fmodel = MODEL_ARCH(quantized=True).to(device)
    fmodel.load_state_dict(qmodel.state_dict())

    train_stat, test_stat = None, None

    for layer_name in qmodel.freeze_layer_iterator():
        flayer = FrozenConv2d(getattr(fmodel, layer_name)).to(device)
        setattr(fmodel, layer_name, flayer)
        optimizer = optim.SGD(fmodel.parameters(), lr=1e-5, momentum=0.9)

        loss = nn.CrossEntropyLoss()
        train_stat, test_stat = train(fmodel, optimizer, train_loader, test_loader, loss,
                                      n_epochs=EP_FREEZE, train_stat=train_stat, test_stat=test_stat,
                                      device=device, use_tqdm=True, print_results=False,
                                      model_update=model_update)
    return fmodel, train_stat, test_stat

def quantization(name, model, in_bins, weight_bins, test_loader, train_loader, symmetric=True,
                 model_update=None, device='cpu'):
    quant_start = time()
    quantized_model = ptq_model(model, in_bins, weight_bins, symmetric, train_loader, device=device)
    save_model(quantized_model, f'{name}_ptq', test_loader, device=device)

    frozen_model, train_stat, test_stat = gradual_freeze_model(quantized_model, train_loader, test_loader, device=device,
                                                               model_update=model_update)
    quant_end = time()
    save_model(frozen_model, f'{name}_quant', test_loader, quant_end - quant_start, device)
    save_statistics(f'{name}_quant', train_stat, test_stat)
    return frozen_model, train_stat, test_stat


transformer_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(9),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
])
transformer_test = transforms.Compose([
        transforms.ToTensor(),
])

DATA_PATH = './data'
BATCH_SIZE = 100

train_set = datasets.CIFAR10(DATA_PATH, train=True, transform=transformer_train, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = datasets.CIFAR10(DATA_PATH, train=False, transform=transformer_test, download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


step_epochs = 50

lr_scale = 0.5
n_steps = 1

to_train_model = MODEL_ARCH()       # model
device = torch.device("cpu")

NAME = "CNN8_f_cov"
feat_cov_reg = 1

optimizer = optim.AdamW(to_train_model.parameters(), lr=2e-3, weight_decay=1e-5)

mean_corrs = []    # method
corrs_l = [[] for _ in to_train_model.conv_layers]
pearson_corr = count_mean_filter_correlation_model(mean_corrs, corrs_l)

fc_reg = feat_cov_loss(feat_cov_reg, [0, 1])

scheduler = optim.lr_scheduler.StepLR(optimizer, step_epochs, lr_scale)
loss = nn.CrossEntropyLoss()

pretrain_start = time()
train_stat, test_stat = train(to_train_model, optimizer, train_loader, test_loader, loss,
                                                n_epochs=step_epochs * n_steps, scheduler=scheduler,
                                                device=device, use_tqdm=True, print_results=False,
                                                model_regularization=fc_reg,
                                                model_update=pearson_corr)
pretrain_end = time()

save_model(to_train_model, f'{NAME}_pretrain', test_loader, pretrain_end - pretrain_start, device=device)
save_statistics(f'{NAME}_pretrain', train_stat, test_stat)

with open(f'{STATISTICS_PATH}/{NAME}_pretrain_corrls_L.json', 'w') as json_file:
    results = {"mean_corrs": mean_corrs, "corrs_l": corrs_l}
    json.dump(results, json_file, indent=2)

# to_train_model = MODEL_ARCH()
# to_train_model.load_state_dict(torch.load("models/CNN8_baseline_pretrain.pt", weights_only=True))
# to_train_model.eval()

mean_corrs = []
corrs_l = [[] for _ in to_train_model.conv_layers]
pearson_corr = count_mean_filter_correlation_model(mean_corrs, corrs_l)

model_quant, train_quant_state, test_quant_state = quantization(NAME, to_train_model, 21, 25, test_loader, train_loader, True,
                                    model_update=pearson_corr)

with open(f'{STATISTICS_PATH}/{NAME}_quant_corrls_L.json', 'w') as json_file:
    results = {"mean_corrs": mean_corrs, "corrs_l": corrs_l}
    json.dump(results, json_file, indent=2)
