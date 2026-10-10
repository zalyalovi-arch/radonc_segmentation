# ============================================================
# postprocess_core.py — per-структурный постпроцессинг масок
# ------------------------------------------------------------
# Движок для GUI-вкладки «Постпроцессинг» и pipeline.py: методы
# применяются к ВЫБРАННОМУ органу (метке), а не ко всей маске.
#
# Ключевая семантика: постпроцессинг никогда не «съедает» другие
# органы. Новые воксели, перекрывающие ЧУЖИЕ метки в базовой
# сегментации, отбрасываются — поэтому fill3d у глаза чинит
# артефактные дыры, но НЕ заливает место линзы (она — отдельная
# метка с большим номером, вложенная в глаз).
#
# Шаблон (JSON): {"template_version": 1, "organs": {"Eye_R": {
#   "label": 25, "methods": [{"method": "closing", "radius": 4}]}}}
# Цепочки применяются к маске органа из БАЗОВОЙ сегментации,
# органы — в порядке возрастания метки (каждая перезаписывает).
# ============================================================
import json
import os

import numpy as np
from scipy import ndimage


def _ball(radius):
    """Сферический структурный элемент без skimage (Euclidean ball)."""
    r = int(radius)
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r + 0.5 * r

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprocess_templates")


# ---------- вспомогательное ----------

def _bbox_slices(mask, margin=6):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    mins, maxs = idx.min(0), idx.max(0) + 1
    shape = mask.shape
    return tuple(slice(max(0, int(m) - margin), min(s, int(M) + margin))
                 for m, M, s in zip(mins, maxs, shape))


def _others_mask(base_seg, label):
    """Чужие органы в базовой сегментации (всё, кроме label)."""
    other = base_seg > 0
    other[base_seg == label] = False
    return other


def _protect_others(result, base_seg, label):
    """Отбрасывает новые воксели, перекрывающие чужие органы."""
    others = _others_mask(base_seg, label)
    return result & ~others


# ---------- методы (маска -> маска, работают в bbox органа) ----------

def m_closing(mask, radius=3):
    st = _ball(int(radius))
    sl = _bbox_slices(mask, margin=int(radius) + 2)
    if sl is None:
        return mask
    crop = mask[sl]
    closed = ndimage.binary_closing(crop, structure=st)
    out = mask.copy()
    out[sl] = closed
    return out


def m_dilate(mask, radius=2):
    st = _ball(int(radius))
    sl = _bbox_slices(mask, margin=int(radius) + 2)
    if sl is None:
        return mask
    out = mask.copy()
    out[sl] = ndimage.binary_dilation(mask[sl], structure=st)
    return out


def m_erode(mask, radius=2):
    st = _ball(int(radius))
    sl = _bbox_slices(mask, margin=int(radius) + 2)
    if sl is None:
        return mask
    out = mask.copy()
    out[sl] = ndimage.binary_erosion(mask[sl], structure=st)
    return out


def m_closing_z(mask, length=3):
    """Closing линейным элементом вдоль Z — закрывает межсрезовые разрывы
    тонких структур (констрикторы), не расширяя в XY."""
    L = int(length)
    st = np.zeros((2 * L + 1, 1, 1), dtype=bool)
    st[:, 0, 0] = True
    sl = _bbox_slices(mask, margin=L + 2)
    if sl is None:
        return mask
    out = mask.copy()
    out[sl] = ndimage.binary_closing(mask[sl], structure=st)
    return out


def m_median(mask, size=3):
    sl = _bbox_slices(mask, margin=int(size))
    if sl is None:
        return mask
    out = mask.copy()
    out[sl] = ndimage.median_filter(mask[sl], size=int(size)) > 0
    return out


def m_keep_largest(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def m_remove_small(mask, min_voxels=50):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.where(sizes >= int(min_voxels))[0]
    return np.isin(lab, keep)


def m_fill_all(mask):
    """Заполняет ВСЕ замкнутые дыры, включая занятые вложенными органами
    (требует allow_overlap в цепочке — иначе защита вернёт дыру на место)."""
    return m_fill3d(mask)


def m_bridge(mask, components=2, thickness=2):
    """Соединяет N крупнейших связных компонент перемычками по ближайшим
    точкам (перешеек щитовидной железы: левая доля + правая доля без
    перешейка). Перемычка утолщается на thickness."""
    lab, n = ndimage.label(mask)
    if n < 2:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = [c for c in np.argsort(sizes)[::-1] if sizes[c] > 0][:int(components)]
    if len(order) < 2:
        return mask
    out = mask.copy()
    prev_pts = None
    for c in order:
        pts = np.argwhere(lab == c)
        if len(pts) > 2000:  # детерминированный прореживающий подсемпл
            pts = pts[:: len(pts) // 2000]
        if prev_pts is not None:
            d = ((prev_pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(d.argmin(), d.shape)
            p0, p1 = prev_pts[i], pts[j]
            steps = int(np.max(np.abs(p1 - p0))) + 1
            t = np.linspace(0, 1, steps)[:, None]
            line = np.round(p0 * (1 - t) + p1 * t).astype(int)
            line = np.unique(line, axis=0)
            if int(thickness) > 1:
                thin = np.zeros(mask.shape, dtype=bool)
                thin[line[:, 0], line[:, 1], line[:, 2]] = True
                thin = ndimage.binary_dilation(thin, structure=_ball(int(thickness)))
                out |= thin
            else:
                out[line[:, 0], line[:, 1], line[:, 2]] = True
        prev_pts = pts
    return out


def m_fill3d(mask):
    sl = _bbox_slices(mask, margin=3)
    if sl is None:
        return mask
    crop = mask[sl]
    filled = ndimage.binary_fill_holes(crop)
    out = mask.copy()
    out[sl] = filled
    return out


def m_fill2d(mask):
    out = mask.copy()
    sl = _bbox_slices(mask, margin=2)
    if sl is None:
        return mask
    z0, z1 = sl[0].start, sl[0].stop
    for z in range(z0, z1):
        out[z] = ndimage.binary_fill_holes(mask[z])
    return out


def m_gaussian(mask, sigma=0.7):
    sm = ndimage.gaussian_filter(mask.astype(np.float32), sigma=float(sigma))
    return sm >= 0.5


METHODS = {
    "closing": m_closing,
    "closing_z": m_closing_z,
    "dilate": m_dilate,
    "erode": m_erode,
    "median": m_median,
    "fill3d": m_fill3d,
    "fill2d": m_fill2d,
    "fill_all": m_fill_all,
    "bridge": m_bridge,
    "gaussian": m_gaussian,
    "keep_largest": m_keep_largest,
    "remove_small": m_remove_small,
}

# какие параметры принимает каждый метод (для GUI)
METHOD_PARAMS = {
    "closing": {"radius": (1, 8, 3)},
    "closing_z": {"length": (1, 10, 3)},
    "dilate": {"radius": (1, 8, 2)},
    "erode": {"radius": (1, 8, 2)},
    "median": {"size": (3, 9, 3)},
    "fill3d": {},
    "fill2d": {},
    "fill_all": {},
    "bridge": {"components": (2, 5, 2), "thickness": (1, 4, 2)},
    "gaussian": {"sigma": (0.3, 2.0, 0.7)},
    "keep_largest": {},
    "remove_small": {"min_voxels": (10, 1000, 50)},
}

# методы, чьи новые воксели могут перекрывать чужие органы (без защиты)
ALLOW_OVERLAP = {"fill_all"}


# ---------- применение ----------

def apply_chain(mask, base_seg, label, chain):
    """Цепочка методов к маске ОДНОГО органа. По умолчанию чужие органы
    защищены (новые воксели поверх чужих меток отбрасываются); методы из
    ALLOW_OVERLAP (fill_all) перекрывают чужие органы намеренно."""
    out = mask
    for step in chain:
        method = step.get("method")
        if method not in METHODS:
            continue
        kwargs = {k: v for k, v in step.items() if k != "method"}
        out = METHODS[method](out, **kwargs)
        out = out.astype(bool)
        if method not in ALLOW_OVERLAP:
            out = mask | (out & ~mask & ~_others_mask(base_seg, label))
    return out.astype(bool)


def apply_template(base_seg, template):
    """Применяет шаблон к полной сегментации. Органы — по возрастанию
    метки, каждая перезаписывает (детерминизм). Возвращает копию."""
    out = base_seg.copy()
    organs = template.get("organs", {})
    for name in sorted(organs, key=lambda n: organs[n].get("label", 0)):
        lbl = int(organs[name].get("label", 0))
        chain = organs[name].get("methods", [])
        if not chain or lbl == 0:
            continue
        mask = base_seg == lbl
        if not mask.any():
            continue
        processed = apply_chain(mask, base_seg, lbl, chain)
        out[mask] = 0
        out[processed] = lbl
    return out


# ---------- метрики ----------

def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return None
    return 2.0 * np.logical_and(a, b).sum() / denom


def hd95(pred, gt, spacing_zyx=None):
    """95-й перцентиль Хаусдорфа (поверхности pred и gt), в единицах spacing.
    Считается в bbox обоих органов (математически эквивалентно полному
    объёму, но на порядки быстрее)."""
    if not pred.any() or not gt.any():
        return None
    both = pred | gt
    sl = _bbox_slices(both, margin=3)
    pred, gt = pred[sl], gt[sl]
    struct = ndimage.generate_binary_structure(3, 1)
    pb = pred & ~ndimage.binary_erosion(pred, structure=struct)
    gb = gt & ~ndimage.binary_erosion(gt, structure=struct)
    if not pb.any() or not gb.any():
        return None
    sampling = spacing_zyx  # (z, y, x) мм
    d_to_gt = ndimage.distance_transform_edt(~gb, sampling=sampling)
    d_to_pred = ndimage.distance_transform_edt(~pb, sampling=sampling)
    dists = np.hstack([d_to_gt[pb], d_to_pred[gb]])
    return float(np.percentile(dists, 95))


# ---------- шаблоны ----------

def save_template(path, template):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    return path


def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_templates():
    if not os.path.isdir(TEMPLATES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(TEMPLATES_DIR) if f.endswith(".json"))


def template_path(name):
    return os.path.join(TEMPLATES_DIR, name + ".json")
