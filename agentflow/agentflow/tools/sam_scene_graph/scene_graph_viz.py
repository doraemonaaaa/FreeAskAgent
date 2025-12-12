"""Matplotlib-based scene graph visualization (static PNG output)."""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional import for matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _mpl_import_error = None
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore
    _mpl_import_error = exc


def _xy_from_obj(obj: dict[str, Any]) -> tuple[float, float]:
    """Return plot coordinates with origin at (0,0) in the center.

    Convention: +x = right, +y = forward (toward top of image).
    If `position` is provided in that convention, use it directly.
    Otherwise, map bbox center from image-normalized [0,1] to centered coords: (x-0.5, 0.5-y).
    """
    pos = obj.get("position") or {}
    if isinstance(pos, dict) and "x" in pos and "y" in pos:
        try:
            return float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        except Exception:
            pass
    bbox = obj.get("bbox") or [0.5, 0.5]
    if len(bbox) >= 2:
        x_c = float(bbox[0])
        y_c = float(bbox[1])
        return x_c - 0.5, 0.5 - y_c
    return 0.0, 0.0


def plot_scene_graph_matplotlib(memory, outfile: str = "scene_graph.png", agent_position: tuple[float, float] | None = (0.0, 0.0)):
    """Render a static scene graph plot with matplotlib and save to a file.

    Falls back to logging when matplotlib is unavailable.
    """
    if plt is None:
        print(f"SceneGraphPlot: matplotlib not available: {_mpl_import_error}")
        return None

    graph = memory.get_scene_graph() or {}
    objs = graph.get("objects", []) or []
    rels = graph.get("relations", []) or []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Scene Graph (local coords)", color="#1c1c1c")
    ax.set_xlabel("x (right)", color="#1c1c1c")
    ax.set_ylabel("y (forward)", color="#1c1c1c")
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, alpha=0.3, color="#9aa0a6")

    xs: list[float] = []
    ys: list[float] = []

    for o in objs:
        x, y = _xy_from_obj(o)
        xs.append(x)
        ys.append(y)
        color = "#0b84a5" if o.get("is_static", True) else "#f39c12"
        ax.scatter([x], [y], c=color, s=70, edgecolors="#1c1c1c", linewidths=0.6, zorder=3)
        label = o.get("label", "obj")
        oid = o.get("object_id", "id")
        ax.text(x + 0.01, y + 0.01, f"{label}\n{oid}", color="#1c1c1c", fontsize=9, zorder=4)

    for r in rels:
        sid, oid = r.get("subject_id"), r.get("object_id")
        src = next((o for o in objs if o.get("object_id") == sid), None)
        dst = next((o for o in objs if o.get("object_id") == oid), None)
        if not src or not dst:
            continue

        sx, sy = _xy_from_obj(src)
        ox, oy = _xy_from_obj(dst)
        ax.annotate(
            "",
            xy=(ox, oy),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4, alpha=0.9),
            zorder=2,
        )
        pred = r.get("predicate", "?")
        ax.text((sx + ox) / 2, (sy + oy) / 2, pred, color="#2f3640", fontsize=9, zorder=5)

    if agent_position is not None:
        ax.scatter([agent_position[0]], [agent_position[1]], marker="s", s=90, c="#2ecc71", edgecolors="#145a32", linewidths=0.8, zorder=4, label="agent")
        ax.text(agent_position[0] + 0.01, agent_position[1] + 0.01, "agent", color="#145a32", fontsize=9, zorder=5)
        xs.append(agent_position[0])
        ys.append(agent_position[1])

    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        span_x = max(1e-3, x_max - x_min)
        span_y = max(1e-3, y_max - y_min)
        pad_x = span_x * 0.2
        pad_y = span_y * 0.2
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
    else:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    fig.savefig(outfile, dpi=150, facecolor="#f8f9fa")
    plt.close(fig)
    print(f"SceneGraphPlot: saved to {outfile}")
    return outfile


__all__ = ["plot_scene_graph_matplotlib"]
