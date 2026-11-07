"""画图的公共样式：dataviz skill 参考色板（palette.md）的浅色版。类别色按固定顺序分配，不循环。"""
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
DASH = (0, (4, 3))


def style_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(which="both", colors=MUTED, labelsize=7.5)  # 对数轴的副刻度标签也要统一


def strength_axis(ax, xs):
    """steering 强度的横轴：symlog（线性区取最小的非零强度），每个网格点一个刻度，0 处画一条竖线。"""
    lin = min(abs(s) for s in xs if s != 0)
    ax.set_xscale("symlog", linthresh=lin, linscale=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s:+g}".replace("-", "−") if s else "0" for s in xs], rotation=90)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)


def line(ax, xs, ys, color, label=None, **kw):
    """2px 折线加 5px 圆点，圆点带一圈背景色描边。"""
    return ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                   markeredgewidth=1.2, label=label, zorder=3, **kw)
