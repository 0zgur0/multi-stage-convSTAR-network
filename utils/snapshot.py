import torch

def save(path, model, optimizer, **kwargs):
    model_state = None
    optimizer_state = None
    if model is not None:
        model_state = model.state_dict()
    if optimizer is not None:
        optimizer_state = optimizer.state_dict()
    torch.save(
        dict(model_state=model_state,
             optimizer_state=optimizer_state,
             **kwargs),
        path
    )


def resume(path, model, optimizer):
    if torch.cuda.is_available():
        snapshot = torch.load(path)
    else:
        snapshot = torch.load(path, map_location="cpu")

    model_state = snapshot.pop('model_state', snapshot)
    optimizer_state = snapshot.pop('optimizer_state', None)

    if model is not None and model_state is not None:
        print("load model")
        model.load_state_dict(model_state)

    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return snapshot