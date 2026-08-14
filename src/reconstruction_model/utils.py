"""General utilities file."""


def count_model_params(model, trainable_only: bool = True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
