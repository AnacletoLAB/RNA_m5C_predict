"""
Run a full hyper‑parameter grid on *one* GPU using ProcessPoolExecutor.

• All heavy libraries (torch, numpy, …) are imported *inside* the worker,
  so the parent stays light and safe.

• CUDA device 0 is selected once and inherited by every child.
  Change GPU_ID below if you like.
"""
import os, json, pickle, datetime, time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

if __name__ == "__main__":
    start = time.time()

    GPU_ID       = 1
    MODEL_NAME   = "Transformer" # "RNN", "1DCNN", "Transformer"
    MAX_WORKERS  = 5
    RUNS_DIR     = Path(f"./runs/{MODEL_NAME}_101_2_missing")
    DATASET_DIR  = Path(__file__).resolve().parents[1] / "dataset"
    FOLDS_DIR    = DATASET_DIR / "folds"



    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


    from utils.search_space import get_search_space, global_search_space, model_search_spaces


    all_cfgs = get_search_space(MODEL_NAME)
    config_missing = False
    if config_missing:
        with open("other_codes/configs_missing.pickle", "rb") as f:
            all_cfgs = pickle.load(f)


    print(f"Total configs to run: {len(all_cfgs)}")


    with (RUNS_DIR / "config_full.json").open("w") as f:
        json.dump(
            {
                "global_params": global_search_space[MODEL_NAME],
                MODEL_NAME: model_search_spaces[MODEL_NAME],
            },
            f, indent=4
        )


def run_one(cfg_idx, g, m, model_name, runs_root, folds_root):
    """
    Train one (global, model) configuration and write its results.
    Returns the run directory for progress tracking.
    """

    import torch, numpy as np
    from utils.init_functions import init_func
    from train_one_run      import training
    from utils.seed         import set_global_seed
    from utils.search_space import model_mapping



    ts = (datetime.datetime.now() + datetime.timedelta(hours=2)).strftime("%d_%m_%Y_%H_%M_%S")
    run_dir = Path(runs_root) / f"run_{cfg_idx}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.json").open("w") as f:
        json.dump({"global_params": g,
                   "model": model_name,
                   "model_params": m}, f, indent=4)

    
    dict_best_results = defaultdict(list)

    for fold_idx in range(1, 6):
        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir()

        with open(Path(folds_root) / f"fold_{fold_idx}_train.pickle", "rb") as f:
            train_dict = pickle.load(f)
        with open(Path(folds_root) / f"fold_{fold_idx}_val.pickle", "rb")   as f:
            val_dict   = pickle.load(f)

        seed = g["base_seed"] + fold_idx
        set_global_seed(seed)

        model = model_mapping(model_name)(**m)
        model.apply(init_func)

        best = training(g, model, train_dict, val_dict, fold_dir, seed)

        for k, v in best.items():
            dict_best_results[k].append(v)

        del model
        torch.cuda.empty_cache()

    # 3‑D. save aggregated metrics
    dict_best_results = dict(dict_best_results)
    averages = {k: float(sum(v) / len(v)) for k, v in dict_best_results.items()}
    with open(run_dir / "results.json",  "w") as f: json.dump(dict_best_results, f)
    with open(run_dir / "averages.json", "w") as f: json.dump(averages,         f)

    return run_dir.as_posix()

if __name__ == "__main__":
    ctx = mp.get_context("spawn") # use spawn to avoid CUDA issues if forking: this way a new cuda context is created for each child
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as pool:
        futures = [
            pool.submit(run_one, idx, g, m,
                        MODEL_NAME, RUNS_DIR.as_posix(), FOLDS_DIR.as_posix())
            for idx, (g, m) in enumerate(all_cfgs)
        ]

        # optional little progress print‑out
        for fut in as_completed(futures):
            print("✅ finished", fut.result())

    print("🎉  All configurations complete.")
    end = time.time()
    minutes = (end - start) / 60
    seconds = (end - start) % 60
    print(f"Total time: {minutes:.0f}m {seconds:.0f}s")
