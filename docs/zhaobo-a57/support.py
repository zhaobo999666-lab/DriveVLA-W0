"""Standalone helpers for zhaobo's A57 tutorial; no training is started here."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(args):
    import numpy as np
    import yaml
    from PIL import Image
    from pyquaternion import Quaternion

    logs, rgb, out = Path(args.logs), Path(args.rgb), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    commands = ["go left", "go straight", "go right", "unknown"]
    for p in sorted(logs.rglob("*.pkl")):
        with p.open("rb") as f:
            frames = pickle.load(f)
        if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
            continue
        if "ego2global_translation" not in frames[0]:
            continue
        poses, names, texts = [], [], []
        for f in frames:
            t = f["ego2global_translation"]
            yaw = Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0]
            poses.append([t[0], t[1], yaw])
            parts = str(f["cams"]["CAM_F0"]["data_path"]).split("/")
            k = parts.index("CAM_F0")
            names.append(f"{parts[k-1]}/{Path(parts[-1]).stem}.npy")
            cmd = np.flatnonzero(np.asarray(f["driving_command"]))
            texts.append(commands[int(cmd[0])] if len(cmd) and int(cmd[0]) < 3 else "unknown")
        poses = np.asarray(poses, dtype=np.float64)
        motion = np.zeros_like(poses)
        for i in range(len(poses)-1):
            dx, dy, dyaw = poses[i+1] - poses[i]
            c, s = math.cos(poses[i, 2]), math.sin(poses[i, 2])
            motion[i] = [c*dx+s*dy, -s*dx+c*dy, (dyaw+math.pi)%(2*math.pi)-math.pi]
        samples = []
        for i in range(5, len(frames)-10):
            if len(frames[i].get("roadblock_ids", [])) == 0:
                continue
            cur, prev = list(range(i-3, i+9)), list(range(i-5, i+7))
            paths = []
            for j in (i, i-2):
                name = Path(names[j])
                paths.append(rgb/name.parent/"CAM_F0"/(name.stem+".jpg"))
            if not all(q.is_file() for q in paths):
                continue
            # Decode both input files now, rather than trusting existence alone.
            for q in paths:
                with Image.open(q) as im:
                    im.verify()
            sample = {"token": str(frames[i]["token"]), "log_name": p.stem,
                      "text": [texts[j] for j in cur], "image": [names[j] for j in cur],
                      "action": motion[cur].copy(),
                      "action_raw": motion[cur].copy(),
                      "pre_1s_text": [texts[j] for j in prev],
                      "pre_1s_image": [names[j] for j in prev],
                      "pre_1s_action": motion[prev].copy()}
            if np.isfinite(sample["action"]).all() and np.isfinite(sample["pre_1s_action"]).all():
                samples.append(sample)
        if samples:
            if p.stem in groups:
                raise ValueError(f"Duplicate raw log name: {p.stem}")
            groups[p.stem] = samples
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need >=2 image-complete logs for an independent holdout. Download more camera shards.")
    cut = min(len(keys)-1, max(1, int(len(keys)*0.8)))
    train_keys, val_keys = keys[:cut], keys[cut:]
    train = [s for k in train_keys for s in groups[k]][:args.max_train]
    val = [s for k in val_keys for s in groups[k]][:args.max_val]
    if not train or not val:
        raise ValueError("Empty train/validation split")
    all_tokens = [s["token"] for s in train+val]
    if len(set(all_tokens)) != len(all_tokens):
        raise ValueError("Duplicate frame tokens; cannot name predictions uniquely")
    a = np.concatenate([s["action"] for s in train])
    low, high = np.percentile(a, [1, 99], axis=0)
    if np.any(high-low < 1e-8):
        raise ValueError("Degenerate action quantiles; expand the dataset")
    stats = {"q01": low.tolist(), "q99": high.tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist()}
    for name, data in [("train", train), ("val", val)]:
        for s in data:
            for field in ("action", "pre_1s_action"):
                s[field] = np.clip(2*(s[field]-low)/(high-low+1e-8)-1, -1, 1).astype(np.float32)
        with (out/f"{name}.pkl").open("wb") as f:
            pickle.dump(data, f)
        (out/f"{name}_tokens.yaml").write_text(yaml.safe_dump({"tokens": [s["token"] for s in data]}), encoding="utf-8")
    write_json(out/"norm_stats.json", {"norm_stats": {"libero": stats}})
    write_json(out/"data_manifest.json", {"train_logs": train_keys, "val_logs": val_keys,
        "train_samples": len(train), "val_samples": len(val), "source_logs": str(logs),
        "rgb_root": str(rgb), "horizon": 8, "interval_seconds": 0.5,
        "warning": "Learning subset; no official navtest or paper-score claim"})
    print(f"Prepared {len(train)} train / {len(val)} validation scenes in {out}")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned-source check failed for: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch(args):
    src = Path(args.repo)/"inference/qwen/inference_action_navsim_vava.py"
    text = src.read_text(encoding="utf-8")
    text = replace_once(text, 'PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)',
                        f'PROJECT_ROOT = {str(Path(args.repo).resolve())!r}')
    text = replace_once(text, "import torch\n", "import torch\nimport torch_npu\n")
    text = replace_once(text, "backend='nccl'", "backend='hccl'")
    text = replace_once(text, 'device = torch.device(f"cuda:{rank}")\n    torch.cuda.set_device(device)',
        'local_rank = int(os.environ.get("LOCAL_RANK", 0))\n    device = torch.device(f"npu:{local_rank}")\n    torch.npu.set_device(device)')
    text = replace_once(text, 'attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
    text = replace_once(text, 'image_processor = AutoProcessor.from_pretrained(args.qwen_hub).image_processor',
        'image_processor = AutoProcessor.from_pretrained(args.qwen_hub).image_processor\n    image_processor.max_pixels = args.max_pixels\n    image_processor.min_pixels = args.min_pixels')
    begin = text.index('        # User historical actions ->')
    end = text.index('        # Build user content with', begin)
    text = text[:begin] + '''        # Training uses no user historical-action segments. The earlier
        # assistant segment contains only motion already observed by time t.
        if args.use_previous_actions:
            raise ValueError("This tutorial matches use_previous_actions=false training")
        pre = np.asarray(scene["pre_1s_action"])[args.cur_frame_idx:args.cur_frame_idx+2]
        pre_ids = action_tokenizer(pre)[0]
        pre_wrapped_action_token_ids = torch.tensor(
            [boa_token_id] + [last_vocab_idx-i for i in pre_ids] + [eoa_token_id], dtype=torch.long)
        if image_tensor is None or pre_image_tensor is None:
            raise ValueError(f"Missing RGB input for scene {scene.get('token', idx)}")

''' + text[end:]
    if 'import numpy as np\n' not in text:
        text = replace_once(text, 'import pickle\n', 'import pickle\nimport numpy as np\n')
    text = replace_once(text, 'action_segments = [pre_user_action_token_ids, pre_wrapped_action_token_ids, user_action_token_ids]',
        'action_segments = [pre_wrapped_action_token_ids]')
    text = replace_once(text, 'action_roles = ["user", "assistant", "user"]', 'action_roles = ["assistant"]')
    text = replace_once(text, 'gt = scene["action"]', 'gt = np.asarray(scene.get("action_raw", scene["action"]))[args.cur_frame_idx:args.cur_frame_idx+args.future_nums]')
    text = replace_once(text, 'gt_denorm = 0.5 * (gt_tensor + 1.0) * (action_high - action_low) + action_low',
        'gt_denorm = gt_tensor if "action_raw" in scene else 0.5 * (gt_tensor + 1.0) * (action_high - action_low) + action_low')
    text = replace_once(text, 'token_name = token_list[idx]',
        'token_name = str(scene["token"])\n        if token_list[idx] != token_name:\n            raise ValueError("token YAML and pickle ordering differ")')
    text = replace_once(text, 'action_tok_ids = [last_vocab_idx - t for t in filtered]',
        'if eoa_token_id not in gen_tokens.tolist() or not filtered:\n            raise ValueError("Incomplete generated action sequence")\n        action_tok_ids = [last_vocab_idx - t for t in filtered]')
    text = replace_once(text, 'pred = torch.tensor(decoded_actions, dtype=torch.float32, device=device)',
        'if np.asarray(decoded_actions).shape != (8, 3) or not np.isfinite(decoded_actions).all():\n            raise ValueError("Invalid decoded actions")\n        pred = torch.tensor(decoded_actions, dtype=torch.float32, device=device)')
    compile(text, str(src), "exec")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}")
    out.write_text(text, encoding="utf-8")
    write_json(out.with_suffix(".provenance.json"), {"source": str(src),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "runtime_status": "Syntax checked only; NPU generation still requires runtime validation"})
    print("Created NPU inference copy:", out)


def report(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, out = Path(args.json_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def positions(a):
        x = y = yaw = 0.
        result = []
        for dx, dy, dt in a:
            x, y = x+math.cos(yaw)*dx-math.sin(yaw)*dy, y+math.sin(yaw)*dx+math.cos(yaw)*dy
            yaw += dt
            result.append([x, y])
        return np.asarray(result)
    rows = []
    for f in sorted(root.glob("*.json")):
        d = json.loads(f.read_text())
        if "action" not in d:
            continue
        if "action_gt_denorm" not in d:
            raise ValueError(f"Missing ground truth: {f}")
        pred, gt = np.asarray(d["action"]), np.asarray(d["action_gt_denorm"])
        if pred.shape != (8, 3) or gt.shape != (8, 3) or not np.isfinite([pred, gt]).all():
            raise ValueError(f"Bad action array: {f}")
        p, g = positions(pred), positions(gt)
        error = np.linalg.norm(p-g, axis=1)
        rows.append({"token": f.stem, "ADE_m": float(error.mean()), "FDE_m": float(error[-1])})
        if len(rows) <= 8:
            fig, ax = plt.subplots()
            ax.plot(np.r_[0, g[:,0]], np.r_[0, g[:,1]], "o-", label="ground truth")
            ax.plot(np.r_[0, p[:,0]], np.r_[0, p[:,1]], "x--", label="prediction")
            ax.set(xlabel="forward x (m)", ylabel="left y (m)", title=f.stem)
            ax.axis("equal"); ax.legend(); fig.tight_layout()
            fig.savefig(out/f"{f.stem}.png", dpi=150); plt.close(fig)
    if not rows:
        raise ValueError("No predictions to evaluate")
    write_json(out/"trajectory_metrics.json", {"samples": len(rows),
        "mean_ADE_m": float(np.mean([r["ADE_m"] for r in rows])),
        "mean_FDE_m": float(np.mean([r["FDE_m"] for r in rows])), "scenes": rows})
    if args.state:
        d = json.loads(Path(args.state).read_text())
        history = [r for r in d.get("log_history", []) if "loss" in r and "step" in r]
        if history:
            fig, ax = plt.subplots()
            ax.plot([r["step"] for r in history], [r["loss"] for r in history])
            ax.set(xlabel="optimizer step", ylabel="training loss")
            fig.tight_layout(); fig.savefig(out/"training_loss.png", dpi=150); plt.close(fig)
    print("Wrote metrics and plots:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--logs", required=True); p.add_argument("--rgb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-train", type=int, default=2048); p.add_argument("--max-val", type=int, default=64)
    p.set_defaults(func=prepare)
    p = sub.add_parser("patch-inference")
    p.add_argument("--repo", required=True); p.add_argument("--out", required=True); p.set_defaults(func=patch)
    p = sub.add_parser("report")
    p.add_argument("--json-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--state"); p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
