# -*- coding: utf-8 -*-
# make_pareto_hw_b.py — Парето-график HW-B (с/патч vs Dice) по данным BENCHMARKS.md.
# Запуск: python scripts/make_pareto_hw_b.py  ->  assets/benchmarks_pareto_hw_b.png
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (метка, с/патч, Dice, цвет, маркер)
PTS = [
    ('Дистиллят INT8 (прод)',            2.40, 0.8387, 'tab:red',   '*'),
    ('Production INT8 (ансамбль 5f)',    2.33, 0.8383, 'tab:red',   'v'),
    ('First-N FP32',                     3.58, 0.8378, 'tab:blue',  'o'),
    ('First-N INT8-веса',                3.53, 0.8377, 'tab:blue',  's'),
    ('First-N INT8 FQ (актив.)',         2.18, 0.8366, 'tab:blue',  'D'),
    ('Prox-обнуление INT8',              2.40, 0.8304, 'tab:gray',  'o'),
    ('Prox-pruned recovered INT8',       2.33, 0.8280, 'tab:gray',  's'),
    ('γ-L2 KDft INT8',                   1.81, 0.8098, 'tab:green', '^'),
    ('L0 dyn INT8 (k20 ε-резка)',        2.48, 0.8304, 'tab:green', 'o'),
    ('L0 static INT8',                   2.52, 0.8304, 'tab:green', 's'),
    ('d2w80 nolens INT8',                1.58, 0.6784, 'tab:brown', '<'),
    ('d2w80_rare INT8',                  1.56, 0.6872, 'tab:brown', '>'),
    ('LADDER7 v8 γ-cut INT8',            2.05, 0.7914, 'tab:purple','P'),
    ('CL-1 v3h γ-cut INT8',              2.35, 0.8385, 'tab:cyan',  'X'),
    ('GRPLSP v11 γ-cut INT8 (НОВЫЙ)',    2.31, 0.8379, 'black',     '*'),
]

fig, ax = plt.subplots(figsize=(11, 7))
front = []
for lbl, t, d, c, m in sorted(PTS, key=lambda p: p[1]):
    if not front or d > front[-1][2]:
        front.append((lbl, t, d))
fx = [t for _, t, _ in front]
fy = [d for _, _, d in front]
ax.plot(fx, fy, '--', color='gray', alpha=0.6, linewidth=1.2, zorder=1)

for lbl, t, d, c, m in PTS:
    big = 'GRPLSP' in lbl
    ax.scatter(t, d, s=260 if big else 90, c=c, marker=m,
               edgecolors='black' if big else 'none',
               linewidths=1.4 if big else 0.5, zorder=3)
    va = 'bottom'
    dy = 0.0012
    if 'GRPLSP' in lbl:
        va, dy = 'top', -0.0022
    ax.annotate(lbl, (t, d + dy), fontsize=7.2, ha='center', va=va, alpha=0.9)

ax.set_xlabel('с/патч (HW-B MSK-D26-9245, stride 0.65, async 6, INT8)', fontsize=10)
ax.set_ylabel('Dice(mean), 10 прод-кейсов', fontsize=10)
ax.set_title('Парето-фронт HW-B — прунинг-линии vs прод-дистиллят (2026-11-12)\n'
             'GRPLSP v11: 21.0% структурной, bin −46% — первая линия, прошедшая прод-гейт 0.8374',
             fontsize=10.5)
ax.set_xlim(1.4, 3.8)
ax.set_ylim(0.66, 0.85)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
fig.tight_layout()
out = r'C:\nnunet_reboot_pyside\assets\benchmarks_pareto_hw_b.png'
fig.savefig(out, dpi=150)
print('saved:', out)
