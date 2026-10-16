"""changes.json, report.md, viewer.html, scene.png, hero.png."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

COLORS = {"removed": "#d62728", "added": "#2ca02c", "moved_from": "#ff9f1c", "moved_to": "#7b2cbf"}


def _r(x, n=3):
    return [round(float(v), n) for v in x]


def _cluster_json(c):
    return {"centroid": _r(c["centroid"]), "bbox_min": _r(c["bbox_min"]), "bbox_max": _r(c["bbox_max"]),
            "n_voxels": c["n_voxels"], "evidence": {k: round(float(v), 2) for k, v in c["evidence"].items()},
            "best_view": c.get("view_own"), "other_view": c.get("view_other"),
            "label_top3": [[l, round(p, 3)] for l, p in c["top3"]], **({"box_px": c["box_px"]} if "box_px" in c else {})}


def _confidence(ch):
    cl = [ch["from"], ch["to"]] if ch["type"] == "moved" else [ch["cluster"]]
    if "confidence" in cl[0]["evidence"]:  # static mode reports its own score
        return round(float(np.mean([c["evidence"]["confidence"] for c in cl])), 3)
    geo = np.mean([min(1.0, c["evidence"]["free_views"] / 6) * c["evidence"]["free_ratio"] for c in cl])
    return round(float(geo), 3)


def changes_json(result):
    items = []
    for i, ch in enumerate(result["changes"]):
        d = {"id": i, "type": ch["type"], "label": ch["label"], "label_confidence": round(ch["label_conf"], 3),
             "geometry_confidence": _confidence(ch), "description": ch["text"]}
        if ch["type"] == "moved":
            d["from"] = _cluster_json(ch["from"])
            d["to"] = _cluster_json(ch["to"])
            d["centroid"] = d["to"]["centroid"]
            d["displacement"] = _r(ch["displacement"])
            d["distance_m"] = round(float(np.linalg.norm(ch["displacement"][:2])), 2)
            d["thumbs"] = {"before": ch["from"].get("thumb_before"), "after": ch["to"].get("thumb_after")}
        else:
            d.update(_cluster_json(ch["cluster"]))
            d["thumbs"] = {"before": ch["cluster"].get("thumb_before"), "after": ch["cluster"].get("thumb_after")}
        items.append(d)
    return {
        "frame": "room frame: z up from the floor, x = viewing direction of the first 'before' frame, "
                 "y = left; units approx. metres",
        "voxel_size": result["voxel"],
        "n_voxels": {"before": int(len(result["cloud_before"]["xyz"])), "after": int(len(result["cloud_after"]["xyz"]))},
        "unobserved_voxels": {"before": int(result["cloud_before"]["unobserved"].sum()),
                              "after": int(result["cloud_after"]["unobserved"].sum())},
        "changes": items,
    }


def write_report(cj, out, before, after):
    L = ["# Detected changes", "", f"Before: `{before}`  ", f"After: `{after}`", ""]
    if not cj["changes"]:
        L.append("No changes detected.")
    for c in cj["changes"]:
        L.append(f"## {c['id']}. {c['type']}: {c['label']}")
        L.append("")
        L.append(c["description"])
        L.append("")
        L.append(f"label confidence {c['label_confidence']:.2f}, geometry confidence {c['geometry_confidence']:.2f}")
        L.append("")
        tb, ta = c["thumbs"]["before"], c["thumbs"]["after"]
        L.append("| before | after |")
        L.append("|---|---|")
        L.append(f"| {'![](' + tb + ')' if tb else '-'} | {'![](' + ta + ')' if ta else '-'} |")
        L.append("")
    u = cj["unobserved_voxels"]
    L.append(f"Unobserved (seen by only one recording, not judged): {u['before']} before / {u['after']} after voxels.")
    (out / "report.md").write_text("\n".join(L) + "\n")


def _scene_points(result, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.concatenate([result["cloud_before"]["xyz"], result["cloud_after"]["xyz"]])
    rgb = np.concatenate([result["cloud_before"]["rgb"], result["cloud_after"]["rgb"]])
    unob = np.concatenate([result["cloud_before"]["unobserved"], result["cloud_after"]["unobserved"]])
    keep = xyz[:, 2] < 2.2  # drop ceiling clutter for readability
    idx = np.nonzero(keep)[0]
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    return xyz[idx], rgb[idx], unob[idx]


def _groups(result):
    """(name, colour, xyz, hover text) for every change cluster."""
    g = []
    for i, ch in enumerate(result["changes"]):
        if ch["type"] == "moved":
            g.append((f"{i} moved {ch['label']} (before)", COLORS["moved_from"], ch["from"]["xyz"], ch["text"]))
            g.append((f"{i} moved {ch['label']} (after)", COLORS["moved_to"], ch["to"]["xyz"], ch["text"]))
        else:
            g.append((f"{i} {ch['type']} {ch['label']}", COLORS[ch["type"]], ch["cluster"]["xyz"], ch["text"]))
    return g


def write_viewer(result, path):
    import plotly.graph_objects as go
    xyz, rgb, unob = _scene_points(result)
    grey = (0.4 * rgb + 0.6 * 170).astype(int)
    fig = go.Figure()
    s = ~unob
    fig.add_trace(go.Scatter3d(x=xyz[s, 0], y=xyz[s, 1], z=xyz[s, 2], mode="markers", name="shared scene",
                               marker=dict(size=1.5, color=[f"rgb({r},{g},{b})" for r, g, b in grey[s]]),
                               hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=xyz[unob, 0], y=xyz[unob, 1], z=xyz[unob, 2], mode="markers",
                               name="unobserved by the other recording",
                               marker=dict(size=1.2, color="#9ecae1", opacity=0.25), hoverinfo="skip"))
    for name, col, p, text in _groups(result):
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                                   marker=dict(size=3, color=col), hovertext=text, hoverinfo="text"))
    for i, ch in enumerate(result["changes"]):
        c = ch["to"]["centroid"] if ch["type"] == "moved" else ch["cluster"]["centroid"]
        fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2] + 0.4], mode="text", text=[f"{i}: {ch['label']}"],
                                   showlegend=False, hovertext=ch["text"], hoverinfo="text"))
        if ch["type"] == "moved":
            a, b = ch["from"]["centroid"], ch["to"]["centroid"]
            fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                                       line=dict(color="black", width=6), showlegend=False, hoverinfo="skip"))
            d = b - a
            fig.add_trace(go.Cone(x=[b[0]], y=[b[1]], z=[b[2]], u=[d[0]], v=[d[1]], w=[d[2]], sizemode="absolute",
                                  sizeref=0.25, anchor="tip", showscale=False, colorscale=[[0, "black"], [1, "black"]],
                                  hoverinfo="skip"))
    cams = np.array([c["center"] for c in result["cameras"]])
    recs = np.array([c["rec"] for c in result["cameras"]])
    for rec, col in (("before", "#555"), ("after", "#999")):
        m = recs == rec
        fig.add_trace(go.Scatter3d(x=cams[m, 0], y=cams[m, 1], z=cams[m, 2], mode="lines+markers",
                                   name=f"{rec} camera path", marker=dict(size=2, color=col),
                                   line=dict(color=col, width=2), hoverinfo="skip"))
    fig.update_layout(title="rayvote: what changed in the room", scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"))
    fig.write_html(path, include_plotlyjs="cdn")


def _draw_3d(ax, result, elev=32, azim=-140):
    xyz, rgb, unob = _scene_points(result, 25000)
    grey = (0.35 * rgb + 0.65 * 175) / 255
    ax.scatter(*xyz[~unob].T, c=grey[~unob], s=0.4, depthshade=False, linewidths=0)
    for name, col, p, _ in _groups(result):
        ax.scatter(*p.T, c=col, s=4, depthshade=False, linewidths=0)
    for i, ch in enumerate(result["changes"]):
        c = ch["to"]["centroid"] if ch["type"] == "moved" else ch["cluster"]["centroid"]
        ax.text(c[0], c[1], c[2] + 0.5, str(i), fontsize=10, weight="bold", ha="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=0.8))
        if ch["type"] == "moved":
            a, b = ch["from"]["centroid"], ch["to"]["centroid"]
            ax.quiver(a[0], a[1], a[2], *(b - a), color="black", linewidth=2, arrow_length_ratio=0.15)
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(0, max(hi[2], 1))
    ax.set_box_aspect(hi - np.array([lo[0], lo[1], 0]) + 1e-3, zoom=1.25)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _wall_rotation(xyz):
    """2D rotation that makes the dominant wall direction axis-aligned (for the plan view only)."""
    from .geometry import normals
    w = xyz[(xyz[:, 2] > 0.5) & (xyz[:, 2] < 2.0)]
    if len(w) < 100:
        return np.eye(2)
    n = normals(w[:: max(1, len(w) // 8000)])
    n = n[np.abs(n[:, 2]) < 0.3]
    ang = np.mod(np.arctan2(n[:, 1], n[:, 0]), np.pi / 2)
    hist, edges = np.histogram(ang, bins=90, range=(0, np.pi / 2))
    a = -(edges[np.argmax(hist)] + np.pi / 180 / 2)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _draw_plan(ax, result):
    """Top-down plan: scene points, camera paths, change footprints, move arrows."""
    xyz, rgb, unob = _scene_points(result, 30000)
    rot = _wall_rotation(xyz)

    def P(p):  # room frame -> plot coords, walls axis-aligned
        q = np.atleast_2d(p)[:, :2] @ rot.T
        return q[:, 0], q[:, 1]
    keep = xyz[:, 2] > 0.1  # drop the floor so furniture and walls read clearly
    ax.scatter(*P(xyz[keep]), c="#b5b5b5", s=0.3, linewidths=0)
    cams = np.array([c["center"] for c in result["cameras"]])
    recs = np.array([c["rec"] for c in result["cameras"]])
    for rec, col in (("before", "#333333"), ("after", "#8a8a8a")):
        m = recs == rec
        ax.plot(*P(cams[m]), "-", color=col, lw=1.2, label=f"{rec} camera path")
    for name, col, p, _ in _groups(result):
        ax.scatter(*P(p), c=col, s=5, linewidths=0)
    for i, ch in enumerate(result["changes"]):
        c = ch["to"]["centroid"] if ch["type"] == "moved" else ch["cluster"]["centroid"]
        if ch["type"] == "moved":
            a, b = ch["from"]["centroid"], ch["to"]["centroid"]
            (ax_, ay_), (bx_, by_) = [np.array(P(x)).ravel() for x in (a, b)]
            ax.annotate("", xy=(bx_, by_), xytext=(ax_, ay_),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="black", shrinkA=4, shrinkB=4))
        cx, cy = np.array(P(c)).ravel()
        ax.text(cx + 0.25, cy + 0.25, str(i), fontsize=10, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=0.8))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="lower left", fontsize=7, frameon=False)


def write_png(result, path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    _draw_3d(ax, result)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_hero(result, out, path):
    ch = result["changes"][:4]
    n = max(len(ch), 1)
    fig = plt.figure(figsize=(14, 9.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 2.1], hspace=0.12)
    top = outer[0].subgridspec(1, n, wspace=0.08)
    for i, c in enumerate(ch):
        sub = top[0, i].subgridspec(1, 2, wspace=0.03)
        if c["type"] == "moved":
            tb, ta = c["from"].get("thumb_before"), c["to"].get("thumb_after")
            col = COLORS["moved_to"]
        else:
            tb, ta = c["cluster"].get("thumb_before"), c["cluster"].get("thumb_after")
            col = COLORS[c["type"]]
        for j, (t, side) in enumerate(((tb, "before"), (ta, "after"))):
            ax = fig.add_subplot(sub[0, j])
            if t:
                ax.imshow(Image.open(out / t))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(col); s.set_linewidth(2.5)
            ax.set_xlabel(side, fontsize=9)
            if j == 0:
                ax.set_title(f"{i}  {c['type']}: {c['label']}", loc="left", fontsize=9.5, weight="bold", color=col)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.02)
    _draw_3d(fig.add_subplot(bot[0, 0], projection="3d"), result)
    _draw_plan(fig.add_subplot(bot[0, 1]), result)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], label=k.replace("_", " "))
               for k in ("removed", "added", "moved_from", "moved_to")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_outputs(result, out, before, after):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cj = changes_json(result)
    (out / "changes.json").write_text(json.dumps(cj, indent=2))
    (out / "cameras.json").write_text(json.dumps(result["cameras"]))
    write_report(cj, out, before, after)
    write_viewer(result, out / "viewer.html")
    write_png(result, out / "scene.png")
    write_hero(result, out, out / "hero.png")
    for c in cj["changes"]:
        print(f"  [{c['id']}] {c['type']:8s} {c['label']:14s} {c['description']}")
