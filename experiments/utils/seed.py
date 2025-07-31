import os, random, numpy as np, torch

def set_global_seed(seed: int):
    """
    Pin python, numpy and torch RNGs so that results are reproducible.
    Call once at the *start* of every training run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic CuDNN (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
