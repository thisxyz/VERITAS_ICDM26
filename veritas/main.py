import argparse
import json


def default_cfg():
    return {
        "data_root": "./data",
        "eps_x": 0.1, "eps_a": 5.0, "kappa": 5,
        "r_star": 0.01, "t": 8, "ff_degree": 3,
        "attack_style": "max", "delta": 30.0,
        "gamma": None, "lam": 1.0,
        "nfr_k": None, "hoa_alpha": 0.5, "hoa_hops": 1,
        "use_nfr": True, "use_hoa": True,
        "input_zscore": True,
        "hidden": 64, "epochs": 400, "lr": 1e-2, "wd": 5e-4,
        "patience": 100, "dropout": 0.5,
        "device": "auto",
    }


def resolve_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mech", default="PM")
    ap.add_argument("--defense", default="veritas",
                    choices=["baseline", "veritas"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--eps-x", type=float, default=0.1)
    ap.add_argument("--eps-a", type=float, default=5.0)
    ap.add_argument("--r-star", type=float, default=0.01)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--attack-style", default=None,
                    choices=["ones", "zeros", "noise", "max", "min", "mean"])
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--no-nfr", action="store_true")
    ap.add_argument("--no-hoa", action="store_true")
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = default_cfg()
    for k in ("dataset", "mech", "defense", "r_star", "gamma", "t", "epochs",
              "attack_style"):
        v = getattr(args, k.replace("-", "_"))
        if v is not None:
            cfg[k] = v
    cfg["eps_x"] = args.eps_x
    cfg["eps_a"] = args.eps_a
    cfg["seed"] = args.seed
    cfg["use_nfr"] = not args.no_nfr
    cfg["use_hoa"] = not args.no_hoa
    cfg["input_zscore"] = not args.no_zscore
    cfg["device"] = resolve_device() if args.device == "auto" else args.device

    from .pipeline import simulate
    results = []
    for s in range(args.seed, args.seed + args.seeds):
        cfg["seed"] = s
        res = simulate(cfg)
        results.append(res)
        print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
