"""Regenerate the data figures of the article (Figs. 12, 13, 15, 16 and 17).

Run from the package root:  python code/make_figures.py
PNG files (300 dpi) are written to figures/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from csh0_model import OUTER_RADIUS_MM, axial_psf, fwhm_3db
from reproduce import (C, DEFS, H, H_ANGLE, ROOT, TH, X, blind, l1, metrics,
                       run, sharpen)

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "xtick.direction": "in", "ytick.direction": "in"})
NAVY, BRICK = (0.08, 0.26, 0.43), (0.80, 0.25, 0.16)
CMAP = LinearSegmentedColormap.from_list("paperprism", np.array(
    [[0.940, 0.935, 0.900], [0.780, 0.780, 0.860], [0.490, 0.620, 0.870], [0.230, 0.690, 0.760],
     [0.450, 0.760, 0.430], [0.880, 0.680, 0.260], [0.950, 0.370, 0.320], [1.000, 0.840, 0.660]]), 256)


# ------------------------------------------------------------------ Fig. 12
def fig12():
    S = np.load(ROOT / "data/simulated_axial_profiles.npz")
    xs, angs, th0 = S["scan_position_mm"], S["defect_angle_deg"], float(S["transducer_angle_deg"])
    a = np.linspace(-45, 45, 9001)
    mk = ["o", "s", "^", "v", "D", "P"]; cm = plt.cm.viridis(np.linspace(0, 0.9, 6))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), dpi=300)
    for ax, key, ap, lbl, tt in [(axs[0], "original_aperture", "original", "(a)", "Original aperture"),
                                 (axs[1], "optimized_aperture", "optimized", "(b)", "Optimized aperture")]:
        model = axial_psf(a, ap)
        ax.plot(a, model, color="k", lw=1.4, label=r"Model $|D|^2$")
        W, R = [], []
        for k, ang in enumerate(angs):
            p = S[key][k].copy()
            if ap == "original" and ang == 130:
                p[10] = 0.5 * (p[9] + p[11])
            s = np.radians(ang - th0) * OUTER_RADIUS_MM
            i = int(np.argmax(p)); y0, y1, y2 = p[i - 1], p[i], p[i + 1]
            xc = xs[i] + 2.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            al = np.degrees(np.arctan((xs - xc) / s))
            ax.plot(al, p / p.max(), mk[k], ms=3, mfc="none", mec=cm[k], mew=0.8, label=f"{ang:.0f}°")
            W.append(2 * np.degrees(np.arctan(fwhm_3db(xs, p) / 2 / s)))
            R.append(np.corrcoef(p / p.max(), np.interp(al, a, model))[0, 1])
        ax.hlines(10 ** (-3 / 20), -14, 14, linestyles=":", color="0.5", lw=0.7)
        ax.set_xlim(-30, 30); ax.set_ylim(0, 1.08); ax.set_xlabel(r"Axial beam angle $\alpha$ (deg)")
        ax.set_title(f"{lbl} {tt}", fontsize=8, loc="left")
        ax.text(0.02, 0.98, f"model {fwhm_3db(a, model):.1f}°\nsim. {np.mean(W):.1f}°\n$r$ = {np.mean(R):.3f}",
                transform=ax.transAxes, va="top", fontsize=6.2)
    # measured axial decay of the slot pairs P1-P4 (probe offset 0-20 mm), optimized prototype
    import json
    pj = ROOT / "outputs/pairs.json"
    if pj.exists():
        ad = json.loads(pj.read_text())["axial_decay"]
        dz = np.array([0, 5, 10, 15, 20.0])
        for pid, r in ad["rows"].items():
            al = np.degrees(np.arctan(dz / r["range_mm"]))
            axs[1].plot(al, [1.0] + r["measured"], "D", ms=2.8, color="k", mec="none",
                        label="Measured P1–P4" if pid == "P1" else None, zorder=5)
        axs[1].text(0.98, 0.98, f"meas. rms dev.\n{ad['rms_difference']:.2f}", transform=axs[1].transAxes,
                    ha="right", va="top", fontsize=6.2)
    axs[0].set_ylabel("Normalized amplitude")
    hd, lb = axs[1].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", bbox_to_anchor=(0.35, 0.0), ncol=7, fontsize=6.2, frameon=False,
               handletextpad=0.3, columnspacing=1.0, title="Defect angle (simulation)", title_fontsize=6.2)
    ax = axs[2]
    ax.plot(H_ANGLE, H, "-", color=NAVY, lw=1.6, label=r"Model $h_\theta$")
    marks = {120: ("o", "#c0392b", "D2"), 200: ("s", "#d68910", "D3"), 280: ("^", "#117a65", "D4")}
    rs, wm = [], []
    for xd, (m, col, name) in marks.items():
        prof = C[:, int(np.argmin(np.abs(X - xd)))]; ip = np.argmax(prof); w = np.abs(TH - TH[ip]) <= 20
        r, sh = max((np.corrcoef(prof[w], np.interp(TH[w] - TH[ip] - s, H_ANGLE, H))[0, 1], s)
                    for s in np.arange(-2, 2.01, 0.1))
        rs.append(r); wm.append(fwhm_3db(TH, prof))
        ax.plot(TH[w] - TH[ip] - sh, prof[w] / prof.max(), m, ms=3, mfc="none", mec=col, mew=0.8,
                label=f"Measured {name}")
    ax.hlines(10 ** (-3 / 20), -16, 16, linestyles=":", color="0.5", lw=0.7)
    ax.set_xlim(-28, 28); ax.set_ylim(0, 1.08); ax.set_xlabel("Angle from peak (deg)")
    ax.set_title("(c) Circumferential PSF", fontsize=8, loc="left")
    ax.text(0.02, 0.98, f"model {fwhm_3db(H_ANGLE, H):.1f}°\nmeas. {np.mean(wm):.1f}°\n"
            f"$r$ = {min(rs):.4f}–\n       {max(rs):.4f}", transform=ax.transAxes, va="top", fontsize=6.2)
    ax.legend(fontsize=5.6, frameon=False, loc="upper right", borderaxespad=0.2, handlelength=1.2, labelspacing=0.3)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "fig12_psf_validation.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 13
def _lift(ax, raw, out, tag, title, w=4.0, ztop=1.0):
    """Original-style 3D view: the measured profile floats above the deconvolved one, linked by its -3 dB extent."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.interpolate import PchipInterpolator
    tp = TH[np.argmax(raw)]; rel = TH - tp
    x = np.arange(-20, 20.001, 0.1)
    rb = np.clip(PchipInterpolator(rel, raw / raw.max())(x), 0, None)
    ra = np.clip(PchipInterpolator(rel, out / out.max())(x), 0, None)
    ax.computed_zorder = False
    for prof, z, zo in ((ra, 0.0, 2), (rb, ztop, 6)):
        quads = [[(x[i], 0, z), (x[i + 1], 0, z), (x[i + 1], w, z), (x[i], w, z)] for i in range(len(x) - 1)]
        ax.add_collection3d(Poly3DCollection(quads, facecolors=CMAP(0.5 * (prof[:-1] + prof[1:])), edgecolors="none",
                                             zorder=zo))
    def edges(prof):
        k = np.where(prof >= 10 ** (-3 / 20))[0]; return x[k[0]], x[k[-1]]
    (b0, b1), (a0, a1) = edges(rb), edges(ra)
    fan = [[(b0, w / 2, ztop), (b1, w / 2, ztop), (a1, w / 2, 0), (a0, w / 2, 0)]]
    ax.add_collection3d(Poly3DCollection(fan, facecolors="0.55", alpha=0.18, edgecolors="0.45", linewidths=0.6, zorder=4))
    for xs in ((b0, a0), (b1, a1)):
        ax.plot(xs, (w / 2, w / 2), (ztop, 0), color="0.45", lw=0.6, zorder=5)
    xc = x[np.argmax(ra)]
    ax.plot((x[np.argmax(rb)], xc), (w / 2, w / 2), (ztop, 0), "k--", lw=0.8, zorder=7)
    ax.plot((b0, b1), (w / 2, w / 2), (ztop, ztop), color=NAVY, lw=1.6, zorder=8)
    ax.text(-20, w, ztop + 0.12, f"Before deconvolution\n−3 dB = {fwhm_3db(TH, raw):.1f}°", fontsize=9.5, ha="left", zorder=9)
    ax.text(-21, w, 0.16, f"After L1 deconvolution\n−3 dB = {fwhm_3db(TH, out):.1f}°", fontsize=9.5, ha="left", zorder=9)
    ax.set_xlim(-20, 20); ax.set_ylim(0, w); ax.set_zlim(0, ztop)
    ax.view_init(elev=24, azim=-112)
    ax.set_box_aspect((2.6, 0.5, 1.1))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False); ax.set_yticks([]); ax.set_zticks([])
    ax.yaxis.line.set_visible(False); ax.zaxis.line.set_visible(False)
    ax.set_xticks(range(-20, 21, 5)); ax.tick_params(axis="x", labelsize=8.5, pad=-1)
    ax.set_xlabel("Circumferential angle from peak (°)", fontsize=9.5, labelpad=7)
    ax.text2D(0.0, 0.88, tag, transform=ax.transAxes, fontsize=12)
    ax.text2D(0.10, 0.88, title, transform=ax.transAxes, fontsize=9.5, color="0.3")


def fig13():
    """Deconvolution of measured profiles with h_theta (D2: strongest, D1: weakest)."""
    fig = plt.figure(figsize=(7.4, 2.6), dpi=300)
    for k, (xd, tag, title) in enumerate([(120, "(a)", "D2 (489 mV)"), (40, "(b)", "D1 (58 mV)")]):
        j = int(np.argmin(np.abs(X - xd))); raw = np.clip(C[:, j], 0, None)
        ax = fig.add_axes([k * 0.5, -0.04, 0.5, 1.12], projection="3d")
        _lift(ax, raw, l1(raw), tag, title)
    fig.savefig(OUT / "fig13_deconvolution_steps.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 15
SIZES = ["8×0.5×0.5", "8×1×2", "8×2×2", "8×2×1", "8×2×0.5"]   # mm


def _stepped_contour(ax, img, level, x, y, **kw):
    """Contour drawn along pixel edges, so each 1° × 5 mm pixel keeps its rectangular shape."""
    k = 20
    up = np.kron(img, np.ones((k, k)))
    dx, dy = x[1] - x[0], y[1] - y[0]
    xu = x[0] - dx / 2 + (np.arange(up.shape[1]) + 0.5) * dx / k
    yu = y[0] - dy / 2 + (np.arange(up.shape[0]) + 0.5) * dy / k
    ax.contour(xu, yu, up, levels=[level], **kw)


def fig15(R, M):
    """Layout, fonts and markers follow the original MATLAB figure (top row: before, bottom row: after)."""
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(CMAP(np.linspace(0, 1, 256))); cm.set_bad(CMAP(0.0))
    tfull = np.arange(60, 141, 1.0)                       # display grid 60-140 deg as in the original figure
    xfull = np.arange(0, 401, 5.0)                        # display grid 0-400 mm as in the original figure
    def pad(img):
        out = np.full((tfull.size, xfull.size), np.nan)
        out[np.ix_(np.isin(tfull, TH), np.isin(xfull, X))] = img
        return out
    xmin, xmax = 0.0, 400.0
    ext = [xfull[0] - 2.5, xfull[-1] + 2.5, tfull[0] - 0.5, tfull[-1] + 0.5]
    sel = (X >= 85) & (X <= 155); xx = X[sel]; fsel = (xfull >= 85) & (xfull <= 155)
    zext = [xx[0] - 2.5, xx[-1] + 2.5, tfull[0] - 0.5, tfull[-1] + 0.5]
    d2 = M[1]
    fs = 10
    fig = plt.figure(figsize=(9.6, 4.66), dpi=300)
    rows = [(C, 500, ("(a)", "(b)", "(c)"), f"SNR {d2['snr_before_dB']:.1f} dB", 0.639),
            (R, 160, ("(d)", "(e)", "(f)"),
             f"SNR {d2['snr_after_dB']:.1f} dB\n(+{d2['snr_gain_dB']:.1f} dB, {10 ** (d2['snr_gain_dB'] / 20):.2f}×)", 0.137)]
    for r, (img, vmax, lbl, txt, y0) in enumerate(rows):
        h = 0.304; full = pad(img)
        # ---- (a)/(d) full C-scan
        ax = fig.add_axes([0.055, y0, 0.40, h]); cax = fig.add_axes([0.46, y0, 0.0085, h])
        im = ax.imshow(np.ma.masked_invalid(full), origin="lower", aspect="auto", cmap=cm, vmin=0, vmax=vmax,
                       extent=ext, interpolation="nearest")
        ax.plot([xmin, xmax], [100, 100], ":", color="k", lw=0.6)
        for k, xd in enumerate(DEFS):
            ax.plot(xd, 100, "o", ms=6.5, mfc="w", mec="k", mew=0.8)
            ax.plot(xd, 100, "+", ms=6.5, color="k", mew=1.0)
            ax.text(xd, 80, f"D{k + 1}", ha="center", va="center", fontsize=fs, fontweight="bold")
            if r == 0:
                ax.text(xd, 126, SIZES[k] + " mm", ha="center", va="center", fontsize=fs - 1, color="0.42")
        ax.add_patch(plt.Rectangle((85, 85), 70, 35, fill=False, ls=(0, (4, 3)), lw=0.9, ec="k"))
        ax.set_xlim(xmin, xmax); ax.set_ylim(60, 140)
        ax.set_xticks(range(0, 401, 50)); ax.set_yticks(range(60, 141, 20))
        ax.tick_params(labelsize=fs); ax.set_xlabel("Axial distance (mm)", fontsize=fs)
        ax.set_ylabel("Circumferential angle (°)", fontsize=fs)
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=fs, direction="in")
        cb.set_ticks(np.linspace(0, vmax, 6) if vmax == 500 else [0, 50, 100, 150])
        ax.text(0.5, -0.33, lbl[0], transform=ax.transAxes, ha="center", va="top", fontsize=fs + 4)
        # ---- (b)/(e) zoom on D2
        bx = fig.add_axes([0.565, y0, 0.125, h]); bcax = fig.add_axes([0.696, y0, 0.0085, h])
        im2 = bx.imshow(np.ma.masked_invalid(full[:, fsel]), origin="lower", aspect="auto", cmap=cm, vmin=0,
                        vmax=vmax, extent=zext, interpolation="nearest")
        bx.plot(120, 100, "+", ms=8, color="k", mew=1.1)
        bx.set_xlim(85, 155); bx.set_ylim(85, 120); bx.set_xticks([100, 120, 140]); bx.set_yticks([90, 100, 110, 120])
        bx.tick_params(labelsize=fs)
        bx.set_xlabel("Axial distance (mm)", fontsize=fs); bx.set_ylabel("Circumferential angle (°)", fontsize=fs)
        bx.text(0.96, 0.04, txt, transform=bx.transAxes, ha="right", va="bottom", fontsize=fs - 1,
                bbox=dict(fc="w", ec="none", pad=2))
        cb2 = fig.colorbar(im2, cax=bcax); cb2.ax.tick_params(labelsize=fs, direction="in")
        cb2.set_ticks(np.linspace(0, vmax, 6) if vmax == 500 else [0, 50, 100, 150])
        cb2.set_label("Amplitude (mV)", fontsize=fs)
        bx.text(0.5, -0.33, lbl[1], transform=bx.transAxes, ha="center", va="top", fontsize=fs + 4)
        # ---- (c)/(f) 50 mV contour along pixel edges
        cx = fig.add_axes([0.835, y0, 0.125, h])
        _stepped_contour(cx, img[:, sel], 50.0, xx, TH, colors="k", linewidths=1.2)
        cx.plot(120, 100, "+", ms=8, color="k", mew=1.1)
        cx.set_xlim(85, 155); cx.set_ylim(85, 120); cx.set_xticks([100, 120, 140]); cx.set_yticks([90, 100, 110, 120])
        cx.tick_params(labelsize=fs)
        cx.set_xlabel("Axial distance (mm)", fontsize=fs); cx.set_ylabel("Circumferential angle (°)", fontsize=fs)
        cx.text(0.97, 0.04, "50 mV contour", transform=cx.transAxes, ha="right", va="bottom", fontsize=fs - 1,
                color="0.5")
        cx.text(0.5, -0.33, lbl[2], transform=cx.transAxes, ha="center", va="top", fontsize=fs + 4)
    fig.savefig(OUT / "fig15_cscan_deconvolution.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 16
def pair_figure_data():
    """Profiles of the measured slot pairs P1-P4 (aligned positions), referenced to the D2 arrival."""
    import json
    import reproduce_pairs as RP
    out = json.loads((ROOT / "outputs/pairs.json").read_text()) if (ROOT / "outputs/pairs.json").exists() else None
    if out is None:
        RP.main(); out = json.loads((ROOT / "outputs/pairs.json").read_text())
    sets = out["sets"]["averaged64_V"]; shift = out["reference_shift_deg"]
    rows = []
    for pid in ("P1", "P2", "P3", "P4"):
        lab = [l for l in sets["raw"]["rows"] if l.startswith(pid + "_z")][0]
        get = lambda m: sets[m]["rows"][lab]
        rows.append(dict(pid=pid, sep=RP.SEPARATION[pid], slots=RP.NOMINAL[pid],
                         raw=np.array(get("raw")["profile_mV"]), l1=np.array(get("envelope_L1_model")["profile_mV"]),
                         coh=np.array(get("coherent_L1_model_cal")["profile_mV"]),
                         ok_raw=get("raw")["resolved"], ok_l1=get("envelope_L1_model")["resolved"],
                         ok_coh=get("coherent_L1_model_cal")["resolved"]))
    return rows, shift


def _lobe_span(x, prof, x0, thr=10 ** (-3 / 20)):
    """-3 dB span of the lobe of `prof` whose maximum is nearest to x0 (prof on grid x)."""
    pk = [i for i in range(1, len(prof) - 1) if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.15]
    i = min(pk, key=lambda k: abs(x[k] - x0)) if pk else int(np.argmax(prof))
    lev = thr * prof[i]
    l = i
    while l > 0 and prof[l] >= lev:
        l -= 1
    r = i
    while r < len(prof) - 1 and prof[r] >= lev:
        r += 1
    return x[l], x[r]


def _pair_panel(ax, r, shift, w=4.0, ztop=1.0):
    """One slot pair in the visual language of Figs. 13 and 17: the raw profile floats above the
    model-PSF restoration; grey wedges join the raw lobe(s) to the restored peaks (a forked wedge
    when the two echoes merge in the raw profile); dashed lines with triangles mark the slots."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.interpolate import PchipInterpolator
    rel = TH - shift - 100.0                         # angle from the pair centre (D2 arrival = 100 deg)
    x = np.arange(-20, 20.001, 0.1)
    fine = lambda p: np.clip(PchipInterpolator(rel, p / p.max())(x), 0, None)
    rb, ra = fine(r["raw"]), fine(r["l1"])
    ax.computed_zorder = False
    for prof, z, zo in ((ra, 0.0, 2), (rb, ztop, 7)):
        quads = [[(x[i], 0, z), (x[i + 1], 0, z), (x[i + 1], w, z), (x[i], w, z)] for i in range(len(x) - 1)]
        ax.add_collection3d(Poly3DCollection(quads, facecolors=CMAP(0.5 * (prof[:-1] + prof[1:])),
                                             edgecolors="none", zorder=zo))
    slots = [c - 100.0 for c in r["slots"]]
    ym = w / 2
    raw_spans = [_lobe_span(x, rb, c) for c in slots]
    res_spans = [_lobe_span(x, ra, c) for c in slots]
    wedge = dict(facecolors="0.55", alpha=0.17, edgecolors="0.5", linewidths=0.5, zorder=4)
    if abs(raw_spans[0][0] - raw_spans[1][0]) < 0.05:          # one raw lobe: forked wedge
        (b0, b1), (l0, l1), (q0, q1) = raw_spans[0], res_spans[0], res_spans[1]
        xm, zn = 0.5 * (l1 + q0), 0.42 * ztop
        poly = [(b0, ym, ztop), (b1, ym, ztop), (q1, ym, 0), (q0, ym, 0), (xm, ym, zn), (l1, ym, 0), (l0, ym, 0)]
        ax.add_collection3d(Poly3DCollection([poly], **wedge))
    else:
        for (b0, b1), (a0, a1) in zip(raw_spans, res_spans):
            ax.add_collection3d(Poly3DCollection([[(b0, ym, ztop), (b1, ym, ztop), (a1, ym, 0), (a0, ym, 0)]], **wedge))
    for b0, b1 in {tuple(sp) for sp in raw_spans}:
        ax.plot((b0, b1), (ym, ym), (ztop, ztop), color=NAVY, lw=1.5, zorder=9)
    for c in slots:
        ax.plot((c, c), (ym, ym), (ztop, 0), "k--", lw=0.7, zorder=8)
        ax.plot([c], [ym], [ztop + 0.075], marker="v", ms=4.2, color="k", mec="none", zorder=12)
    st = lambda ok: ("resolved", BRICK) if ok else ("merged", NAVY)
    (wb, cb), (wa, ca) = st(r["ok_raw"]), st(r["ok_l1"])
    ax.text(22.5, ym, ztop, wb, fontsize=8, ha="left", va="center", color=cb, fontweight="bold", zorder=11)
    ax.text(22.5, ym, 0.0, wa, fontsize=8, ha="left", va="center", color=ca, fontweight="bold", zorder=11)
    ax.set_xlim(-20, 20); ax.set_ylim(0, w); ax.set_zlim(0, ztop)
    ax.view_init(elev=24, azim=-112)
    ax.set_box_aspect((2.6, 0.5, 1.1), zoom=1.08)
    ax.set_facecolor((1, 1, 1, 0))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False); ax.set_yticks([]); ax.set_zticks([])
    ax.yaxis.line.set_visible(False); ax.zaxis.line.set_visible(False)
    ax.set_xticks(range(-20, 21, 5)); ax.tick_params(axis="x", labelsize=7, pad=-2)
    ax.set_xlabel("Angle from pair centre (°)", fontsize=8, labelpad=3)


def fig16():
    """Measured separability of the slot pairs P1-P4 drawn on magnified sectors of the pipe wall (one row):
    outer ring = raw C-scan profile, inner ring = profile after L1 deconvolution with h_theta,
    black notches = machined slots, dashed radial lines = slot directions."""
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon, Wedge
    from scipy.interpolate import PchipInterpolator
    for f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf", "texgyretermes-italic.otf"):
        try:
            font_manager.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/" + f)
        except Exception:
            pass
    rc = {"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"],
          "mathtext.fontset": "stix"}
    anchors = np.array([[1.000, 1.000, 1.000], [0.780, 0.780, 0.860], [0.490, 0.620, 0.870], [0.230, 0.690, 0.760],
                        [0.450, 0.760, 0.430], [0.880, 0.680, 0.260], [0.950, 0.370, 0.320], [1.000, 0.840, 0.660]])
    cmap = LinearSegmentedColormap.from_list("paperprism_white", anchors, 256)
    rows, shift = pair_figure_data()
    rel = TH - shift - 100.0
    x = np.arange(-20, 20.001, 0.1)
    fine = lambda p: np.clip(PchipInterpolator(rel, p / p.max())(x), 0, None)
    R0, R1 = 0.965, 1.035            # pipe wall (thickness exaggerated)
    a0, a1 = 1.07, 1.30              # outer ring: raw C-scan
    b0, b1 = 0.70, 0.93              # inner ring: after L1 deconvolution
    mag = 2.4                        # angular magnification of the +-20 deg window
    ang = lambda deg: np.radians(90 - deg * mag)
    INK = NAVY
    XL, YL = (-1.02, 1.02), (0.17, 1.66)
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.4, 2.15), dpi=300)
        x_left, x_right, gap = 0.115, 0.905, 0.008
        w = (x_right - x_left - 3 * gap) / 4
        axs = [fig.add_axes([x_left + k * (w + gap), 0.0, w, 1.0]) for k in range(4)]
        for k, (ax, r) in enumerate(zip(axs, rows)):
            for prof, (ra, rb_) in ((fine(r["raw"]), (a0, a1)), (fine(r["l1"]), (b0, b1))):
                for i in range(len(x) - 1):
                    c = cmap(0.5 * (prof[i] + prof[i + 1]))
                    ax.add_patch(Wedge((0, 0), rb_, 90 - x[i + 1] * mag, 90 - x[i] * mag, width=rb_ - ra,
                                       fc=c, ec=c, lw=0.3))
                ax.add_patch(Wedge((0, 0), rb_, 90 - 20 * mag, 90 + 20 * mag, width=rb_ - ra, fc="none", ec=INK, lw=0.6))
            ax.add_patch(Wedge((0, 0), R1, 90 - 20 * mag, 90 + 20 * mag, width=R1 - R0, fc="white", ec=INK, lw=0.9))
            for t in range(-20, 21, 5):
                th = ang(t); ro = a1 + (0.035 if t % 10 == 0 else 0.02)
                ax.plot([a1 * np.cos(th), ro * np.cos(th)], [a1 * np.sin(th), ro * np.sin(th)], color=INK, lw=0.6)
                if t % 10 == 0 and abs(t) < 20:
                    ax.text(1.40 * np.cos(th), 1.40 * np.sin(th), f"{t:+d}°" if t else "0°", ha="center", va="center",
                            fontsize=6.5, rotation=-t * mag)
            for c in r["slots"]:
                th = ang(c - 100); hw = np.radians(1.3 * mag) / 2; d = 0.05
                pts = [(R1 * np.cos(th - hw), R1 * np.sin(th - hw)), (R1 * np.cos(th + hw), R1 * np.sin(th + hw)),
                       ((R1 - d) * np.cos(th + hw), (R1 - d) * np.sin(th + hw)), ((R1 - d) * np.cos(th - hw), (R1 - d) * np.sin(th - hw))]
                ax.add_patch(Polygon(pts, fc="k", ec="k", lw=0.4, zorder=6))
                ax.plot([b0 * np.cos(th), a1 * np.cos(th)], [b0 * np.sin(th), a1 * np.sin(th)], color="k", lw=0.7,
                        ls=(0, (2.5, 2)), zorder=7)
            st = lambda ok: ("resolved", BRICK) if ok else ("merged", NAVY)
            (w1, c1), (w2, c2) = st(r["ok_raw"]), st(r["ok_l1"])
            # status: centred in the concave space below the inner ring, raw above L1
            ax.text(0, 0.375, f"Raw: {w1}", ha="center", va="center", fontsize=7.5, color=c1, fontweight="bold")
            ax.text(0, 0.235, f"L1: {w2}", ha="center", va="center", fontsize=7.5, color=c2, fontweight="bold")
            # title: centred above the sector
            ax.text(0, 1.585, f"({chr(97 + k)})  {r['pid']},  {r['sep']}°", ha="center", va="center", fontsize=9)
            ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.axis("off")
        # left key: ring names with horizontal leaders to the left ends of panel (a); balances the colour bar
        th = ang(-20)
        for rr, ty, txt in (((a0 + a1) / 2, 1.00, "Raw C-scan"), ((R0 + R1) / 2, 0.70, "Pipe wall"),
                            ((b0 + b1) / 2, 0.40, "After L1")):
            px, py = rr * np.cos(th), rr * np.sin(th)
            axs[0].annotate(txt, xy=(px, py), xytext=(-2.20, ty), textcoords="data", fontsize=7.5, ha="left",
                            va="center", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color="k", lw=0.5, shrinkA=2, shrinkB=1))
        cax = fig.add_axes([0.928, 0.22, 0.011, 0.56])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax)
        cb.set_label("Normalized\namplitude", fontsize=7); cb.ax.tick_params(labelsize=6.5)
        cb.set_ticks([0, 0.5, 1.0])
        fig.savefig(OUT / "fig16_separability.png", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


# ------------------------------------------------------------------ Fig. 17
METHODS = ["Model PSF", "Generic Gaussian", "Blind", "Sharpen only"]
MCOLS = ["#c0392b", "#148f77", "#6c4f9c", "#d68910"]


def _strips(ax, raw, profiles, t0=75.0, t1=135.0, w=2.6, gap=4.2, ztop=1.25):
    """Original-style 3D view: the raw profile floats above, the four reconstructions lie below,
    and translucent fans link the -3 dB extent of the raw profile to that of each reconstruction."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.interpolate import PchipInterpolator
    x = np.arange(t0, t1 + 0.001, 0.1)

    def fine(p):
        q = np.clip(PchipInterpolator(TH, np.clip(p, 0, None))(x), 0, None); return q / q.max()

    def strip(prof, y0, z, zo):
        quads = [[(x[i], y0, z), (x[i + 1], y0, z), (x[i + 1], y0 + w, z), (x[i], y0 + w, z)] for i in range(len(x) - 1)]
        ax.add_collection3d(Poly3DCollection(quads, facecolors=CMAP(0.5 * (prof[:-1] + prof[1:])), edgecolors="none",
                                             zorder=zo))

    def segments(prof):
        above = prof >= 10 ** (-3 / 20); segs = []; k = 0
        while k < len(x):
            if above[k]:
                m = k
                while m + 1 < len(x) and above[m + 1]:
                    m += 1
                segs.append((x[k], x[m])); k = m + 1
            else:
                k += 1
        return segs

    ax.computed_zorder = False
    yr = 1.5 * gap                                   # raw strip floats above the middle of the stack
    rb = fine(raw)
    (r0, r1), = [max(segments(rb), key=lambda s: s[1] - s[0])]
    for k in (3, 2, 1, 0):                           # back to front
        pk = fine(profiles[k]); y0 = k * gap
        strip(pk, y0, 0.0, 2)
        for (a0, a1) in segments(pk):
            fan = [[(r0, yr + w / 2, ztop), (r1, yr + w / 2, ztop), (a1, y0 + w / 2, 0), (a0, y0 + w / 2, 0)]]
            ax.add_collection3d(Poly3DCollection(fan, facecolors=MCOLS[k], alpha=0.16, edgecolors="none", zorder=4))
            ax.plot((a0, a1), (y0 + w / 2, y0 + w / 2), (0.003, 0.003), color=MCOLS[k], lw=2.2, zorder=5)
        ax.text(t0 - 1.5, y0 + w / 2, 0, METHODS[k], ha="right", va="center", fontsize=9.5,
                color=MCOLS[0] if k == 0 else "k", zorder=10)
    strip(rb, yr, ztop, 6)
    tp = x[np.argmax(rb)]
    ax.plot((tp, tp), (yr + w / 2, w / 2), (ztop, 0), "k--", lw=0.8, zorder=7)
    ax.plot((tp, tp), (w / 2, 3 * gap + w / 2), (0.004, 0.004), "k--", lw=0.6, zorder=7)
    ax.plot((r0, r1), (yr + w / 2, yr + w / 2), (ztop, ztop), color=NAVY, lw=1.8, zorder=8)
    ax.text(t0 - 4, yr + w, ztop + 0.22, f"Raw D2\n−3 dB = {fwhm_3db(TH, raw):.1f}°", fontsize=9.5, ha="right", zorder=9)
    ax.set_xlim(t0, t1); ax.set_ylim(0, 3 * gap + w); ax.set_zlim(0, ztop)
    ax.view_init(elev=24, azim=-112)
    ax.set_box_aspect((2.4, 1.3, 0.9))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False)
    ax.set_yticks([]); ax.set_zticks([]); ax.yaxis.line.set_visible(False); ax.zaxis.line.set_visible(False)
    ax.set_xticks(range(80, 131, 10)); ax.tick_params(axis="x", labelsize=9, pad=-1)
    ax.set_xlabel("Circumferential angle (°)", fontsize=10, labelpad=6)


def fig17(results):
    names = ["Model PSF $h_\\theta$", "Generic Gaussian", "Blind", "Sharpen only"]
    j = int(np.argmin(np.abs(X - 120)))
    fig = plt.figure(figsize=(7.2, 6.0), dpi=300)
    ax = fig.add_axes([0.04, 0.30, 0.86, 0.70], projection="3d")
    _strips(ax, C[:, j], [results[n][0][:, j] for n in names])
    for k, (txt, col) in enumerate([("Model PSF: single narrow peak ✓", MCOLS[0]),
                                    ("Generic Gaussian: false double peak", "k"),
                                    ("Blind: stays broad", "k"), ("Sharpen only: stays broad", "k")]):
        fig.text(0.07, 0.965 - k * 0.032, txt, fontsize=10.5, color=col,
                 fontfamily=["Times New Roman", "Liberation Serif", "DejaVu Sans"])
    fig.text(0.015, 0.965, "(a)", fontsize=12)
    cax = fig.add_axes([0.90, 0.55, 0.016, 0.36])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), cax=cax)
    cb.set_label("Normalized image amplitude", fontsize=10); cb.ax.tick_params(labelsize=9)
    panels = [("secondary_peak", "Secondary-peak level", "lower is better", "{:.2f}"),
              ("peak_shift_deg", "Peak shift (deg)", "lower is better", "{:.2f}"),
              ("width_after_deg", "Restored width (deg)", "hatched: double peak", "{:.2f}"),
              ("snr_gain_dB", "SNR gain (dB)", "higher is better", "{:+.1f}")]
    for k, (key, title, hint, fmt) in enumerate(panels):
        bx = fig.add_axes([0.075 + k * 0.237, 0.04, 0.19, 0.27])
        vals = [[r[key] for r in results[n][1]] for n in names]
        means = [float(np.mean(v)) for v in vals]
        bars = bx.bar(range(4), means, color=MCOLS, width=0.66)
        if key == "width_after_deg":
            bars[1].set_hatch("////"); bars[1].set_edgecolor("w")
        for i, v in enumerate(vals):
            bx.plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, ".", color="0.2", ms=2.5)
        top = max(max(v) for v in vals); bot = min(0.0, min(min(v) for v in vals))
        for i, m in enumerate(means):
            yv = max(vals[i]) if m >= 0 else min(vals[i])
            bx.text(i, yv + 0.05 * (top - bot) * (1 if m >= 0 else -2.2), fmt.format(m), ha="center", va="bottom",
                    fontsize=8, color=MCOLS[i], fontweight="bold")
        bx.set_ylim(bot - 0.08 * (top - bot) * (bot < 0) * 3, top * 1.45)
        bx.axhline(0, color="k", lw=0.6)
        bx.set_xticks([]); bx.tick_params(labelsize=7)
        bx.set_title(title, fontsize=9.5)
        bx.text(0.97, 0.96, hint, transform=bx.transAxes, ha="right", va="top", fontsize=8, color="0.5")
        if key == "snr_gain_dB":
            bx.text(0.12, 0.96, "✓", transform=bx.transAxes, ha="center", va="top", fontsize=13, color=MCOLS[0],
                    fontfamily="DejaVu Sans")
            hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MCOLS]
            bx.legend(hs, METHODS, loc="upper right", bbox_to_anchor=(1.02, 0.86), frameon=False,
                      handlelength=0.7, handleheight=0.8, handletextpad=0.4, labelspacing=0.35, labelcolor=MCOLS,
                      prop={"weight": "bold", "size": 6.0})
    fig.text(0.015, 0.36, "(b)", fontsize=12)
    fig.savefig(OUT / "fig17_method_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fig12(); print("Fig. 12 done")
    fig13(); print("Fig. 13 done")
    methods = {"Model PSF $h_\\theta$": l1, "Generic Gaussian": lambda p: l1(p, "gauss"),
               "Blind": blind, "Sharpen only": sharpen}
    results = {}
    for n, m in methods.items():
        R = run(m); results[n] = (R, metrics(R))
    fig15(*results["Model PSF $h_\\theta$"]); print("Fig. 15 done")
    fig16(); print("Fig. 16 done")
    fig17(results); print("Fig. 17 done")
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
