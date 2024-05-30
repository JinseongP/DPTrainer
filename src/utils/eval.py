import torch

def get_accuracy(model, data_loader, atk=None, n_limit=1e10, device=None):
    model = model.eval()
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)

    correct = 0
    total = 0

    for images, labels in data_loader: 
        X = images.to(device)
        Y = labels.to(device)

        if atk:
            X = atk(X, Y)

        with torch.no_grad():
            pre = model(X)

        try:
            _, pre = torch.max(pre.data, 1)
        except:
            pre = pre[0]
            _, pre = torch.max(pre.data, 1)

        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break
    return (100 * float(correct) / total)
