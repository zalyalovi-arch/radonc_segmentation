# -*- coding: utf-8 -*-
"""
Per-stage Effective Rank + класс-пробы для студента xtiny80_d2w (training-free, CPU).

Вопрос: существуют ли на нужных масштабах линейные направления для 32 классов
(особенно мёртвой пятёрки OpticNerve R/L, Lens R/L, Cochlea_L) на случайной
инициализации, и отличается ли stride-1 стебель (1 блок, 32 канала) от живых классов.

Метод:
1. Сеть собирается как в nnU-Net: get_network_from_plans с arch_kwargs ResEncM,
   где features_per_stage/blocks заменены на xtiny80_d2w (планы наследуются,
   ядра/strides те же). 3 seeds: 42/123/456 (инициализация), анкор-патчи общие.
2. Данные: 10 локальных GT-кейсов (validation/headandneck, уже в таргет
   spacing), CTNormalization-аппроксимация (глобальные перцентили/mean/std).
   Кейсы обрабатываются потоково, в RAM держатся только патчи.
3. Патчи 56x192x192: анкор на вокселе класса (5 мёртвых + 3 контроля) +
   случайные. До 3 кейсов x 2 патча на класс.
4. Per-stage ER: грам-аккумулятор A·A^T по позициям (глобально и внутри маски
   класса), собственные числа грама -> ER = exp(-sum p ln p).
5. Класс-пробы: на каждой стадии логистическая регрессия "класс vs фон" на
   клетках сетки стадии, leave-one-case-out, AUC; шанс-контроль на перемешанных
   метках.
"""
import gc
import json
import os
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom as ndzoom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

# ── Константы ────────────────────────────────────────────────────────────────
ROOT = r"C:\nnunet_reboot_pyside\validation\headandneck\GT"
OUT_DIR = r"C:\nnunet_reboot_pyside\perstage_er"
PATCH = (56, 192, 192)          # (Z, Y, X)
TARGET_SPACING = (2.0, 1.171875, 1.171875)
JITTER = 0.15
SEEDS = [42, 123, 456]
N_ANCHOR_CASES = 3              # кейсов на класс
PATCHES_PER_CASE = 2
MIN_POS_IN_PATCH = 10           # минимум позитивных вокселей класса в патче
N_RANDOM_PATCHES = 5
GRAM_POS_PER_PATCH = 30000      # сэмпл позиций для глобального грама
MAX_NEG_TOTAL = 40000
NEG_PER_POS = 20
HU_CLIP = (-1024.0, 3071.0)
ANCHOR_SEED = 999               # патчи общие для всех seed'ов (парное сравнение)

# 0-based индексы форграунда (= значение метки - 1)
DEAD = {"OpticNerve_R": 13, "OpticNerve_L": 14, "Lens_R": 24, "Lens_L": 26, "Cochlea_L": 30}
CTRL = {"Cochlea_R": 29, "Trachea": 4, "Mandible": 22}
CLASSES = {**DEAD, **CTRL}      # name -> fg index (label value = idx+1)

XTINY_FEATURES = [32, 64, 80, 80, 80, 80]
XTINY_BLOCKS = [1, 2, 2, 3, 3, 3]


def er_from_gram(G):
    """ER из грам-матрицы CxC признаков (по сингулярным числам A)."""
    w = np.linalg.eigvalsh(G.astype(np.float64))
    s = np.sqrt(np.clip(w, 0.0, None))
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = s / tot
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def load_dataset_json():
    with open(os.path.join(os.path.dirname(ROOT), "predictions", "dataset.json")) as f:
        dj = json.load(f)
    mapping, idx = {}, 0
    for name in dj["labels"]:
        if name == "background":
            continue
        mapping[name] = idx
        idx += 1
    return mapping


def crop_patch(vol, center, patch):
    """Кроп каждого тома по ВСЕМ 3 осям, рефлекс-паддинг у краёв."""
    out = []
    for v in vol:
        piece = v
        for axis, (c, p) in enumerate(zip(center, patch)):
            size = piece.shape[axis]
            half = p // 2
            lo = int(np.clip(c - half, -half, size - 1))
            hi = lo + p
            pad_lo, pad_hi = max(0, -lo), max(0, hi - size)
            sl = [slice(None)] * piece.ndim
            sl[axis] = slice(max(0, lo), min(size, hi))
            piece = piece[tuple(sl)]
            pads = [(pad_lo, pad_hi) if ax == axis else (0, 0) for ax in range(piece.ndim)]
            piece = np.pad(piece, pads, mode="reflect")
            fin = [slice(0, p) if ax == axis else slice(None) for ax in range(piece.ndim)]
            piece = piece[tuple(fin)]
        out.append(piece)
    return out


def collect_patches():
    """Потоковая загрузка кейсов: считаем статы нормализации и сразу режем патчи."""
    img_dir = os.path.join(ROOT, "imagesTr")
    lbl_dir = os.path.join(ROOT, "labelsTr")
    rng = np.random.default_rng(ANCHOR_SEED)
    stat_samples = []
    patches = []
    per_class_cases = {c: 0 for c in CLASSES}
    n_rand = 0
    files = sorted(os.listdir(lbl_dir))
    for k, f in enumerate(files):
        case = f.replace(".nii.gz", "")
        img = nib.load(os.path.join(img_dir, case + "_0000.nii.gz"))
        lbl = nib.load(os.path.join(lbl_dir, f))
        zooms = img.header.get_zooms()[:3]
        a = np.asarray(img.dataobj, dtype=np.float32)
        b = np.asarray(lbl.dataobj)
        order = np.argsort(zooms)[::-1]          # ось 0 = самая грубая (срезы)
        a = np.transpose(a, order)
        b = np.transpose(b, order)
        fac = np.array(zooms)[order] / np.array(TARGET_SPACING)
        if np.any(np.abs(fac - 1) > 1e-3):
            a = ndzoom(a, fac, order=1)
            b = ndzoom(b, fac, order=0).astype(np.uint8)
        a = np.clip(a, *HU_CLIP)
        stat_samples.append(a.ravel()[:: max(1, a.size // 150000)])

        # анкор-патчи по классам (общие для всех seed'ов; лимит N_ANCHOR_CASES кейсов)
        for name, fg in CLASSES.items():
            if per_class_cases[name] >= N_ANCHOR_CASES:
                continue
            lab_val = fg + 1
            pos = np.argwhere(b == lab_val)
            if pos.shape[0] < 50:
                continue
            got = 0
            for _ in range(8):
                if got >= PATCHES_PER_CASE:
                    break
                anchor = pos[rng.integers(pos.shape[0])]
                jit = (rng.random(3) * 2 - 1) * (np.array(PATCH) * JITTER)
                center = np.round(anchor + jit).astype(int)
                img_p, lbl_p = crop_patch([a, b], center, PATCH)
                assert img_p.shape == PATCH and lbl_p.shape == PATCH, \
                    f"кроп: {img_p.shape} != {PATCH}"
                if int((lbl_p == lab_val).sum()) < MIN_POS_IN_PATCH:
                    continue
                patches.append({"case": case, "class": name,
                                "img": np.round(img_p).astype(np.int16), "lbl": lbl_p})
                got += 1
                if got == PATCHES_PER_CASE:
                    per_class_cases[name] += 1

        if n_rand < N_RANDOM_PATCHES and k < 5:
            center = np.array([rng.integers(p, s - p) if s > 2 * p else s // 2
                               for p, s in zip(PATCH, a.shape)])
            img_p, lbl_p = crop_patch([a, b], center, PATCH)
            patches.append({"case": case, "class": "_random",
                            "img": np.round(img_p).astype(np.int16), "lbl": lbl_p})
            n_rand += 1
        del a, b
        gc.collect()
        print(f"  {case}: ок (патчей всего {len(patches)})", flush=True)

    print("Патчей по классам (кейсов):",
          {c: per_class_cases[c] for c in CLASSES})

    v = np.concatenate(stat_samples)
    p_lo, p_hi = np.percentile(v, [0.5, 99.5])
    vv = v[(v >= p_lo) & (v <= p_hi)]
    stats = (float(p_lo), float(p_hi), float(vv.mean()), float(vv.std()))
    return patches, stats


def build_network(seed):
    plans_path = os.path.join(os.path.dirname(ROOT), "predictions", "plans.json")
    plans = json.load(open(plans_path))
    arch = plans["configurations"]["3d_fullres"]["architecture"]
    kw = dict(arch["arch_kwargs"])
    kw["features_per_stage"] = XTINY_FEATURES
    kw["n_blocks_per_stage"] = XTINY_BLOCKS
    torch.manual_seed(seed)
    np.random.seed(seed)
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    net = get_network_from_plans(arch["network_class_name"], kw, arch["_kw_requires_import"],
                                 input_channels=1, output_channels=32, allow_init=True,
                                 deep_supervision=True)
    net.eval()
    return net


def gram_accumulate(acc, feat):
    """feat: (C, P) -> обновить аккумулятор (сумма грамов, счётчик)."""
    A = feat.astype(np.float32)
    G = A @ A.T
    if acc["G"] is None:
        acc["G"] = G
        acc["n"] = A.shape[1]
    else:
        acc["G"] += G
        acc["n"] += A.shape[1]


def pool_mask(mask, stride):
    """Маска (Z,Y,X) -> клетки стадии (доля позитивных вокселей в клетке)."""
    t = torch.from_numpy(mask[None, None].astype(np.float32))
    p = F.avg_pool3d(t, kernel_size=stride, stride=stride)
    return p[0, 0].numpy()


def pick_patch_mode(stats):
    """Пробуем полный патч 56x192x192; при OOM (im2col stage-1 ~1.8 ГБ) — кроп 56x112x112."""
    net = build_network(1)
    x = torch.randn(1, 1, *PATCH)
    try:
        net.encoder(x)
        del net
        print("Режим патча: полный 56x192x192")
        return PATCH
    except RuntimeError as e:
        if "not enough memory" not in str(e):
            raise
    del net
    gc.collect()
    alt = (56, 112, 112)
    net = build_network(1)
    net.encoder(torch.randn(1, 1, *alt))
    del net
    gc.collect()
    print(f"Режим патча: кроп {alt} (RAM-лимит, im2col stage-1)")
    return alt


def run_seed(seed, patches, stats, use_patch):
    net = build_network(seed)
    enc = net.encoder
    cum, s = [], np.ones(3, dtype=int)
    for st in enc.strides:
        s = s * np.array(st)
        cum.append(tuple(int(v) for v in s))
    n_stages = len(cum)
    if use_patch != PATCH:
        off = [(P - p) // 2 for P, p in zip(PATCH, use_patch)]
        sl = tuple(slice(o, o + p) for o, p in zip(off, use_patch))
    else:
        sl = None

    glob = [{"G": None, "n": 0} for _ in range(n_stages)]
    cls_gram = {c: [{"G": None, "n": 0} for _ in range(n_stages)] for c in CLASSES}
    probe = {c: {i: {"pos": [], "neg": [], "groups": []} for i in range(n_stages)}
             for c in CLASSES}

    n, t0 = 0, time.time()
    for p in patches:
        img_p, lab = p["img"], p["lbl"]
        if sl is not None:
            img_p, lab = img_p[sl], lab[sl]
        x = (img_p.astype(np.float32) - stats[2]) / stats[3]
        x = torch.from_numpy(x[None, None].copy())
        skips = enc(x)
        for i, feat in enumerate(skips):
            f = feat[0] if feat.dim() == 5 else feat
            f = f.numpy()  # (C, Z, Y, X)
            C = f.shape[0]
            flat = f.reshape(C, -1)
            P = flat.shape[1]
            idx = np.random.default_rng(seed * 1000 + n * 97 + i).choice(
                P, size=min(GRAM_POS_PER_PATCH, P), replace=False)
            gram_accumulate(glob[i], flat[:, idx])

            if p["class"] in CLASSES:
                fg_idx = CLASSES[p["class"]] + 1
                m = pool_mask((lab == fg_idx).astype(np.float32), cum[i])
                pos_cells = np.argwhere(m >= 0.5)
                grid = f.shape[1:]
                assert m.shape == tuple(grid), f"grid {m.shape} != {tuple(grid)}"
                if pos_cells.size:
                    lin = (pos_cells[:, 0].astype(np.int64) * grid[1]
                           + pos_cells[:, 1]) * grid[2] + pos_cells[:, 2]
                    posf = flat[:, lin]
                    gram_accumulate(cls_gram[p["class"]][i], posf)
                    rec = probe[p["class"]][i]
                    rec["pos"].append(posf.T.astype(np.float32))
                    rec["groups"] += [p["case"]] * posf.shape[1]
                    neg_pool = np.argwhere(m < 0.01)
                    if neg_pool.size:
                        k = min(neg_pool.shape[0], NEG_PER_POS * posf.shape[1],
                                MAX_NEG_TOTAL // (n_stages * len(CLASSES)))
                        sel = np.random.default_rng(seed * 77 + n + i).choice(
                            neg_pool.shape[0], size=k, replace=False)
                        lin_n = (neg_pool[sel][:, 0].astype(np.int64) * grid[1]
                                 + neg_pool[sel][:, 1]) * grid[2] + neg_pool[sel][:, 2]
                        negf = flat[:, lin_n]
                        rec["neg"].append(negf.T.astype(np.float32))
                        rec["groups"] += [p["case"]] * negf.shape[1]
        del skips, feat, f, flat
        n += 1
        if n % 10 == 0:
            gc.collect()
            print(f"    seed {seed}: {n}/{len(patches)} патчей, {time.time()-t0:.0f} c", flush=True)

    res = {"seed": seed,
           "er_global": [er_from_gram(g["G"]) if g["G"] is not None else None for g in glob],
           "er_class": {c: [er_from_gram(g["G"]) if g["G"] is not None and g["n"] >= 5 else None
                            for g in cls_gram[c]] for c in CLASSES},
           "probe": {c: {i: None for i in range(n_stages)} for c in CLASSES}}

    # Логистические пробы leave-one-case-out
    for c in CLASSES:
        for i in range(n_stages):
            rec = probe[c][i]
            if not rec["pos"]:
                continue
            Xp = np.concatenate(rec["pos"])
            Xn = np.concatenate(rec["neg"]) if rec["neg"] else np.zeros((0, 0), np.float32)
            groups = np.array(rec["groups"])
            why = None
            if Xp.shape[0] < 20:
                why = f"n_pos={Xp.shape[0]}<20"
            elif Xn.shape[0] < 100:
                why = f"n_neg={Xn.shape[0]}<100"
            elif len(set(groups.tolist())) < 2:
                why = f"groups<2 ({sorted(set(groups.tolist()))})"
            if why:
                print(f"    probe skip {c} s{i}: {why}", flush=True)
                continue
            try:
                np.savez_compressed(os.path.join(OUT_DIR, f"probe_{c}_s{i}.npz"),
                                    X=np.concatenate([Xp, Xn]),
                                    y=np.concatenate([np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])]),
                                    groups=groups)
            except OSError as e:
                print(f"    npz save fail {c} s{i}: {e}", flush=True)
            X = np.concatenate([Xp, Xn])
            y = np.concatenate([np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])])
            mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
            X = (X - mu) / sd
            aucs, aucs_shuf = [], []
            logo = LeaveOneGroupOut()
            uniq = list(set(groups.tolist()))
            for tr, te in logo.split(X, y, groups=groups):
                if y[te].sum() == 0 or (1 - y[te]).sum() == 0 or len(uniq) < 2:
                    continue
                if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
                    continue
                try:
                    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
                    lr.fit(X[tr], y[tr])
                    aucs.append(roc_auc_score(y[te], lr.decision_function(X[te])))
                    rs = np.random.default_rng(seed + i).permutation(y[tr].shape[0])
                    lr2 = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
                    lr2.fit(X[tr], y[tr][rs])
                    aucs_shuf.append(roc_auc_score(y[te], lr2.decision_function(X[te])))
                except ValueError:
                    continue
            if aucs:
                res["probe"][c][i] = {"auc": float(np.mean(aucs)),
                                      "auc_chance": float(np.mean(aucs_shuf)) if aucs_shuf else None,
                                      "n_pos": int(Xp.shape[0]), "n_neg": int(Xn.shape[0]),
                                      "folds": len(aucs)}
    del net, enc
    gc.collect()
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("Сбор патчей и статистики нормализации...")
    mapping = load_dataset_json()
    for name, idx in CLASSES.items():
        assert mapping.get(name) == idx, f"маппинг {name}: {mapping.get(name)} != {idx}"
    print("Маппинг классов подтверждён")
    patches, stats = collect_patches()
    by_class = {}
    for p in patches:
        by_class[p["class"]] = by_class.get(p["class"], 0) + 1
    print(f"Патчи: {by_class}")
    print(f"Нормализация: clip [{stats[0]:.0f}, {stats[1]:.0f}], mean {stats[2]:.1f}, std {stats[3]:.1f}")

    all_res = []
    use_patch = pick_patch_mode(stats)
    for seed in SEEDS:
        print(f"\n=== SEED {seed} ===", flush=True)
        r = run_seed(seed, patches, stats, use_patch)
        all_res.append(r)
        print(f"  ER global: {[None if v is None else round(v,1) for v in r['er_global']]}")
        for c in CLASSES:
            a0 = r["probe"][c][0]
            print(f"  {c:14s} ER: {[None if v is None else round(v,1) for v in r['er_class'][c]]} "
                  f"| AUC@stage0: {None if a0 is None else round(a0['auc'],3)}")
    del patches
    gc.collect()

    with open(os.path.join(OUT_DIR, "perstage_er_results.json"), "w") as f:
        json.dump(all_res, f, indent=1)
    print(f"\nГотово за {time.time()-t0:.0f} c -> perstage_er_results.json")


if __name__ == "__main__":
    main()
