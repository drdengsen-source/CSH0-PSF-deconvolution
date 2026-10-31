"""Reproduce the numerical results of the article from the packaged data.

Run from the package root:  python code/reproduce.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from csh0_model import (APERTURES_MM, OUTER_RADIUS_MM, axial_psf, circumferential_psf,
                        directivity, fwhm_3db)
from deconvolution import deconvolve_l1

ROOT = Path(__file__).resolve().parents[1]
GAIN_B = 0.013080802295779018   # propagation-gain constant G_B (1/deg), Eq. (3.5)
REG_REL = 0.11                  # lambda = 0.11 * max(guarded profile)
GUARD = 20                      # mirror guard band (samples = deg)

# ---------------------------------------------------------------- data
E = np.load(ROOT / "data/experimental_cscan.npz")
C, TH, X, DEFS = E["cscan_mV"], E["theta_deg"], E["axial_position_mm"], E["defect_axial_position_mm"]
BAND = np.abs(TH - 101) <= 25                                   # 76-126 deg
BG_COLS = [j for j, x in enumerate(X) if np.min(np.abs(x - DEFS)) >= 20]
H_ANGLE, H = circumferential_psf(355.9e3)


def kernel(offset_deg, kind="model", fwhm=None):
    if kind == "model":
        h = np.interp(offset_deg, H_ANGLE, H, left=0, right=0)
    else:  # generic Gaussian
        h = np.exp(-0.5 * (offset_deg / (fwhm / 2.3548)) ** 2)
    return h / h.max()


def _prep(profile):
    p = np.clip(np.asarray(profile, float), 0, None)
    tp = TH[np.argmax(p)]
    corr = p * np.exp(GAIN_B * (TH - tp))                         # Eq. (3.6)
    g = GUARD
    padded = np.concatenate([corr[g:0:-1], corr, corr[-2:-g - 2:-1]])
    off = np.arange(len(padded), dtype=float) - len(padded) // 2
    return corr, padded, off, tp


def _finish(core, tp):
    return np.clip(core * np.exp(-GAIN_B * (TH - tp)), 0, None)


def l1(profile, kind="model", rel=REG_REL):
    corr, padded, off, tp = _prep(profile)
    fw = fwhm_3db(TH, np.clip(profile, 0, None)) if kind == "gauss" else None
    o = deconvolve_l1(padded, kernel(off, kind, fw), rel * padded.max(), iterations=500)
    return _finish(o[GUARD:GUARD + len(corr)], tp)


def blind(profile, outer=20, inner=5):
    corr, padded, off, tp = _prep(profile)
    F, IF = np.fft.fft, lambda v: np.real(np.fft.ifft(v))
    h = kernel(off, "gauss", fwhm_3db(TH, corr)); h /= h.sum()
    o = np.full_like(padded, padded.mean())
    for _ in range(outer):
        for _ in range(inner):
            Hf = F(np.fft.ifftshift(h)); est = IF(F(o) * Hf)
            o = o * IF(F(padded / np.maximum(est, 1e-12)) * np.conj(Hf))
        for _ in range(inner):
            O = F(o); hs = np.fft.ifftshift(h); est = IF(F(hs) * O)
            hs = hs * IF(F(padded / np.maximum(est, 1e-12)) * np.conj(O)); hs /= hs.sum()
            h = np.fft.fftshift(hs)
    return _finish(o[GUARD:GUARD + len(corr)], tp)


def sharpen(profile, amount=1.0):
    corr, padded, off, tp = _prep(profile)
    k = kernel(off, "gauss", fwhm_3db(TH, corr) / 2); k /= k.sum()
    blur = np.real(np.fft.ifft(np.fft.fft(padded) * np.fft.fft(np.fft.ifftshift(k))))
    out = np.clip(padded + amount * (padded - blur), 0, None)
    return _finish(out[GUARD:GUARD + len(corr)], tp)


def metrics(R):
    nb = np.std(np.concatenate([C[BAND, j] for j in BG_COLS]))
    na = np.std(np.concatenate([R[BAND, j] for j in BG_COLS]))

    def centroid(p):
        i = int(np.argmax(p))
        y0, y1, y2 = p[i - 1], p[i], p[i + 1]
        den = y0 - 2 * y1 + y2
        return TH[i] + (0.5 * (y0 - y2) / den if den else 0.0)

    rows = []
    for xd in DEFS:
        j = int(np.argmin(np.abs(X - xd))); b, a = C[:, j], R[:, j]
        sb, sa = 20 * np.log10(b.max() / nb), 20 * np.log10(a.max() / na)
        ip = int(np.argmax(a))
        lm = [k for k in range(1, len(a) - 1) if a[k] >= a[k - 1] and a[k] > a[k + 1]
              and BAND[k] and abs(TH[k] - TH[ip]) > 2]
        rows.append(dict(defect_mm=float(xd), peak_before_mV=b.max(), peak_after_mV=a.max(),
                         snr_before_dB=sb, snr_after_dB=sa, snr_gain_dB=sa - sb,
                         width_before_deg=fwhm_3db(TH, b), width_after_deg=fwhm_3db(TH, a),
                         peak_shift_deg=abs(centroid(a) - centroid(b)),
                         secondary_peak=max([a[k] for k in lm], default=0.0) / a.max()))
    return rows


def run(method):
    return np.column_stack([method(C[:, j]) for j in range(C.shape[1])])


# ----------------------------------------------------- separability (Fig. 16)
def _resolved(prof, c1, c2, tol=2.5, dip=0.81):
    lm = [k for k in range(1, len(prof) - 1) if prof[k] >= prof[k - 1] and prof[k] > prof[k + 1]
          and prof[k] > 0.2 * prof.max()]
    k1 = [k for k in lm if abs(TH[k] - c1) <= tol]; k2 = [k for k in lm if abs(TH[k] - c2) <= tol]
    if not k1 or not k2:
        return False
    i1 = max(k1, key=lambda k: prof[k]); i2 = max(k2, key=lambda k: prof[k])
    lo, hi = sorted((i1, i2))
    return prof[lo:hi + 1].min() <= dip * min(prof[i1], prof[i2])


def separability(xa, xb, ctr=104.0):
    def col(xd):
        c = np.clip(C[:, int(np.argmin(np.abs(X - xd)))], 0, None); return c, TH[np.argmax(c)]
    seps = np.arange(2, 26, 1.0); before, after = [], []
    (a, ta), (b, tb) = col(xa), col(xb)
    for sep in seps:
        c1, c2 = ctr - sep / 2, ctr + sep / 2
        prof = (np.interp(TH, TH - (ta - c1), a, left=0, right=0)
                + np.interp(TH, TH - (tb - c2), b, left=0, right=0))
        before.append(_resolved(prof, c1, c2)); after.append(_resolved(l1(prof), c1, c2))

    def limit(r):
        for i in range(len(seps)):
            if all(r[i:]):
                return float(seps[i])
        return float("nan")
    return limit(before), limit(after)


# ----------------------------------------------- axial PSF validation (Fig. 12(a),(b))
def axial_validation():
    S = np.load(ROOT / "data/simulated_axial_profiles.npz")
    xs, angs, th0 = S["scan_position_mm"], S["defect_angle_deg"], float(S["transducer_angle_deg"])
    a = np.linspace(-45, 45, 9001); out = {}
    for key, ap in [("original_aperture", "original"), ("optimized_aperture", "optimized")]:
        widths, corrs = [], []
        for k, ang in enumerate(angs):
            p = S[key][k].copy()
            if ap == "original" and ang == 130:   # single-sample artefact, see README
                p[10] = 0.5 * (p[9] + p[11])
            s = np.radians(ang - th0) * OUTER_RADIUS_MM
            i = int(np.argmax(p)); y0, y1, y2 = p[i - 1], p[i], p[i + 1]
            xc = xs[i] + 2.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            L = fwhm_3db(xs, p)
            widths.append(2 * np.degrees(np.arctan(L / 2 / s)))
            corrs.append(np.corrcoef(p / p.max(), np.interp(np.degrees(np.arctan((xs - xc) / s)),
                                                            a, axial_psf(a, ap)))[0, 1])
        model = fwhm_3db(a, axial_psf(a, ap))
        out[ap] = dict(simulated_width_deg=float(np.mean(widths)), model_width_deg=model,
                       difference_pct=abs(np.mean(widths) - model) / model * 100,
                       shape_correlation=float(np.mean(corrs)))
    return out


def main():
    a = np.linspace(-90, 90, 18001)
    summary = {"one_way_beamwidth_deg": {ap: fwhm_3db(a, directivity(np.deg2rad(a), *rc))
                                         for ap, rc in APERTURES_MM.items()},
               "axial_psf_validation": axial_validation()}
    h340 = circumferential_psf(340e3)
    summary["circumferential_psf_width_deg"] = {"355.9 kHz": fwhm_3db(H_ANGLE, H),
                                                "340 kHz": fwhm_3db(*h340)}
    corrs = []
    for xd in (120, 200, 280):
        col = C[:, int(np.argmin(np.abs(X - xd)))]; ip = np.argmax(col); w = np.abs(TH - TH[ip]) <= 20
        corrs.append(max(np.corrcoef(col[w], np.interp(TH[w] - TH[ip] - s, H_ANGLE, H))[0, 1]
                         for s in np.arange(-2, 2.01, 0.1)))
    summary["circumferential_psf_correlation_D2_D4"] = corrs
    methods = {"model_psf": l1, "gaussian": lambda p: l1(p, "gauss"), "blind": blind, "sharpen": sharpen}
    tables = {name: metrics(run(m)) for name, m in methods.items()}
    summary["separability_limits_deg"] = {f"{xa}+{xb} mm": separability(xa, xb)
                                          for xa, xb in [(120, 200), (120, 280), (200, 280), (280, 360)]}
    summary["four_method_means"] = {n: {k: float(np.mean([r[k] for r in t])) for k in
                                        ("snr_gain_dB", "width_after_deg", "peak_shift_deg", "secondary_peak")}
                                    for n, t in tables.items()}
    (ROOT / "outputs").mkdir(exist_ok=True)
    json.dump({"summary": summary, "table7_table8": tables["model_psf"]},
              open(ROOT / "outputs/results.json", "w"), indent=1, default=float)

    print("One-way -3 dB beamwidth (deg):", {k: round(v, 1) for k, v in summary["one_way_beamwidth_deg"].items()})
    for ap, v in summary["axial_psf_validation"].items():
        print(f"Axial PSF, {ap}: simulated {v['simulated_width_deg']:.1f} deg vs |D|^2 {v['model_width_deg']:.1f} deg "
              f"({v['difference_pct']:.1f}%), r = {v['shape_correlation']:.3f}")
    print("h_theta width (deg):", {k: round(v, 2) for k, v in summary["circumferential_psf_width_deg"].items()},
          "| r(D2-D4) =", [round(float(c), 4) for c in corrs])
    print("\nDefect  SNR before  SNR after  gain   width before  width after  shift")
    for r in tables["model_psf"]:
        print(f"{r['defect_mm']:5.0f}  {r['snr_before_dB']:9.2f}  {r['snr_after_dB']:9.2f}  {r['snr_gain_dB']:+5.2f}"
              f"  {r['width_before_deg']:11.2f}  {r['width_after_deg']:11.2f}  {r['peak_shift_deg']:5.2f}")
    print("\nSeparability limits before/after (deg):", summary["separability_limits_deg"])
    print("Four-method means:")
    for n, v in summary["four_method_means"].items():
        print(f"  {n:10s} gain {v['snr_gain_dB']:+.1f} dB, width {v['width_after_deg']:.2f} deg, "
              f"shift {v['peak_shift_deg']:.2f} deg, secondary peak {v['secondary_peak']:.2f}")
    print("\nResults written to outputs/results.json")


if __name__ == "__main__":
    main()
