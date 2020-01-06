import torch
from torch.utils import data
import numpy as np
import math
import time


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("USING GPU:")
        print(torch.cuda.get_device_name(torch.device('cuda')))

        return [torch.device('cuda'), torch.cuda.get_device_name(torch.device('cuda'))]
    else:
        print("NO GPU AVAILABLE, USING CPU:")
        return [torch.device('cpu'), None]


def to_device(data, device, print_flag):
    """Move tensor(s) to the chosen device"""
    if print_flag:
        print(f"Moving a tensor to device ({device})")
    if isinstance(data, (list,tuple)):
        return [to_device(x, device, print_flag) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device, False) # automatically pushes to device

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def split_indices(n, val_pct):
    val_size = int(val_pct * n)  # first get size of the val set
    idxs = np.random.permutation(n)  # get a random group
    # pick the first (val_size) indices and make them val set
    return idxs[val_size:], idxs[:val_size]


def accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)
    return torch.sum(predictions == labels).item() / len(labels)


def step_decay(epoch, lr_init, drop, epochs_drop):
    """args
        - epoch = current epoch
        - lr_init = initial learning rate
        - drop = drop amount
        - epochs_drop = amount of epochs until a drop
    """
    lr = lr_init * math.pow(drop, math.floor(1+epoch/epochs_drop))
    return lr

def loss_batch(model, loss_function, xb, yb, opt=None, metric=None):
    # first calculate loss
    predictions = model(xb)

    loss = loss_function(predictions, yb)

    if opt is not None: # CHECKING FOR OPTIMIZER 
        # compute the gradients and back prop
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        # compute the metric
        metric_result = metric(predictions, yb)

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_function, valid_data, metric=None):
    with torch.no_grad():
        # pass the batches through
        results = [loss_batch(model, loss_function, xb, yb, metric=metric) for xb, yb in valid_data]

        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        total_loss = np.sum(np.multiply(losses, nums))
        avg_loss = total_loss / total
        avg_metric = None
        if metric is not None:
            total_metric = np.sum(np.multiply(metrics, nums))
            avg_metric = total_metric / total

        return avg_loss, total, avg_metric


def fit(epochs, lr, model, loss_function, train_data, validation_data, metric=None, opt_fn=None, lr_mod=0.0):
    lr_init = lr
    for epoch in range(epochs):
        if opt_fn is None: opt_fn=torch.optim.SGD

        # training
        for xb, yb in train_data:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9)
            loss, _, _ = loss_batch(model, loss_function, xb, yb, opt)


        # evaluate
        result = evaluate(model, loss_function, validation_data, metric)
        validation_loss, total, validation_metric = result

        if metric is None:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {validation_loss}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {validation_loss}, {metric.__name__}: {validation_metric}')
        lr = step_decay(epoch, lr_init, .75, 10) # update the learning rate


def predict_image_with_timing(image, model):
    xb = image.unsqueeze(0) # insert dim at head 
    elapsed_time = 0.0
    yb = model(xb)
    _, predictions = torch.max(yb, dim=1)
    elapsed_time += time.process_time()
    return [(predictions[0].item()), elapsed_time]

def predict_image(image, model):
    xb = image.unsqueeze(0) # insert dim at head
    yb = model(xb)
    _, predictions = torch.max(yb, dim=1)
    return (predictions[0].item())