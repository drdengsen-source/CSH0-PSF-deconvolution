"""Measured two-slot separability (pairs P1-P4) and measured axial decay.

Run from the package root:  python code/reproduce_pairs.py
Writes outputs/pairs.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.signal import hilbert

import coherent as C
import csh0_model as M
import reproduce as R

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "data/supplementary_ascans.npz").exists():          # full record set (single + averaged)
    S = np.load(ROOT / "data/supplementary_ascans.npz")
    SETS = {"averaged64_V": S["averaged64_V"], "single_shot_V": S["single_shot_V"]}
else:                                                            # public minimal set: processed signal, 60-150 deg
    S = np.load(ROOT / "data/slot_pairs_processed.npz")
    SETS = {"averaged64_V": S["analytic_V"]}
T, LAB = S["time_s"], list(S["label"])
TH = C.GRID_DEG
BAND = (TH >= 76) & (TH <= 126)
NOMINAL = {"P1": (89, 111), "P2": (92, 108), "P3": (95, 105), "P4": (97, 103)}
SEPARATION = {"P1": 22, "P2": 16, "P3": 10, "P4": 6}
KAPPAS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
TOL, DIP = 2.5, 0.81                    # resolution criterion of Section 4.2 (unchanged)
TRANSDUCER_DEG = 28.64788976            # theta_0 of Section 3.2


def ref_peak_deg(trace):
    th = C.theta_deg(T)
    e = np.abs(C.analytic_signal(trace))
    m = (th > 85) & (th < 120)
    return float(th[m][np.argmax(e[m])])


def secondary(p):
    i = int(np.argmax(p))
    lm = [k for k in range(1, len(p) - 1) if p[k] >= p[k - 1] and p[k] > p[k + 1]
          and BAND[k] and abs(TH[k] - TH[i]) > 2]
    return max([p[k] for k in lm], default=0.0) / p.max()


def assess(p, c1, c2):
    """Resolution decision and peak/valley measures for one restored profile."""
    lm = [k for k in range(1, len(p) - 1) if p[k] >= p[k - 1] and p[k] > p[k + 1] and p[k] > 0.2 * p.max()]
    k1 = [k for k in lm if abs(TH[k] - c1) <= TOL]
    k2 = [k for k in lm if abs(TH[k] - c2) <= TOL]
    d = dict(resolved=bool(R._resolved(p, c1, c2)))
    if k1 and k2:
        i1 = max(k1, key=lambda k: p[k]); i2 = max(k2, key=lambda k: p[k])
        d.update(peak1_deg=float(TH[i1]), peak2_deg=float(TH[i2]),
                 valley_ratio=float(p[min(i1, i2):max(i1, i2) + 1].min() / min(p[i1], p[i2])))
    return d


def limit(seps, ok):
    o = np.argsort(seps)
    seps, ok = np.asarray(seps)[o], np.asarray(ok)[o]
    for i in range(len(seps)):
        if ok[i:].all():
            return float(seps[i])
    return float("nan")


def methods(scale, kappa):
    env = lambda x: C.envelope_profile(x, T)
    return {
        "raw": env,
        "envelope_L1_model": lambda x: R.l1(env(x)),
        "envelope_L1_gaussian": lambda x: R.l1(env(x), kind="gauss"),
        "blind_RL": lambda x: R.blind(env(x)),
        "unsharp": lambda x: R.sharpen(env(x)),
        "coherent_L1_model": lambda x: C.deconvolve_coherent(x, T, 1.0, kappa),
        "coherent_L1_model_cal": lambda x: C.deconvolve_coherent(x, T, scale, kappa),
    }


def axial_decay():
    """Peak amplitude versus axial offset, normalised to the aligned position, against D^2."""
    th = C.theta_deg(T)
    a = np.linspace(-40, 40, 8001)
    rows = {}
    for pid in NOMINAL:
        base = LAB.index([l for l in LAB if l.startswith(pid + "_z")][0])
        for key in ("averaged64_V",):
            e0 = np.abs(C.analytic_signal(SETS[key][base])) * 1e3
            m = (th > 80) & (th < 125)
            p0 = e0[m].max()
            vals = []
            for off in (5, 10, 15, 20):
                e = np.abs(C.analytic_signal(SETS[key][LAB.index(f"{pid}_offset{off:02d}mm")])) * 1e3
                vals.append(float(e[m].max() / p0))
        theta_d = np.mean(NOMINAL[pid])
        s = M.OUTER_RADIUS_MM * np.radians(theta_d - TRANSDUCER_DEG)
        h = M.axial_psf(a, "optimized")
        pred = [float(np.interp(np.degrees(np.arctan(dz / s)), a, h)) for dz in (5, 10, 15, 20)]
        rows[pid] = dict(range_mm=float(s), measured=vals, predicted=pred)
    meas = np.array([r["measured"] for r in rows.values()])
    pred = np.array([r["predicted"] for r in rows.values()])
    return dict(rows=rows, mean_measured=meas.mean(0).tolist(), mean_predicted=pred.mean(0).tolist(),
                rms_difference=float(np.sqrt(np.mean((meas - pred) ** 2))),
                max_abs_difference=float(np.abs(meas - pred).max()))


def main():
    ref = LAB.index("D2_ref_z120")
    shift = ref_peak_deg(SETS["averaged64_V"][ref]) - 100.0       # arrival-time reference: D2 at 100 deg
    scale = C.calibrated_scale(SETS["averaged64_V"][ref], T)
    sec = {k: float(secondary(C.deconvolve_coherent(SETS["averaged64_V"][ref], T, scale, k))) for k in KAPPAS}
    kappa = min(k for k in KAPPAS if all(sec[j] < 0.2 for j in KAPPAS if j >= k))
    out = dict(reference_shift_deg=shift, wavelet_scale=scale, kappa_nominal=kappa,
               d2_secondary_by_kappa=sec, sets={})
    traces = [i for i, l in enumerate(LAB) if l.startswith("P")]
    for key in SETS:
        res = {}
        for name, f in methods(scale, kappa).items():
            rows = {}
            for i in traces:
                pid = LAB[i][:2]
                c1, c2 = (v + shift for v in NOMINAL[pid])
                p = f(SETS[key][i])
                rows[LAB[i]] = dict(pair=pid, separation_deg=SEPARATION[pid], **assess(p, c1, c2),
                                    profile_mV=p.round(3).tolist())
            aligned = [l for l in rows if "_z" in l]
            res[name] = dict(rows=rows, min_resolvable_aligned_deg=limit(
                [rows[l]["separation_deg"] for l in aligned], [rows[l]["resolved"] for l in aligned]))
        out["sets"][key] = res
    sens = {}
    for k in KAPPAS:
        ok = {}
        for i in traces:
            pid = LAB[i][:2]
            c1, c2 = (v + shift for v in NOMINAL[pid])
            ok[LAB[i]] = bool(R._resolved(C.deconvolve_coherent(SETS["averaged64_V"][i], T, scale, k), c1, c2))
        sens[str(k)] = ok
    out["coherent_kappa_sensitivity"] = sens
    out["axial_decay"] = axial_decay()
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/pairs.json").write_text(json.dumps(out, indent=1))

    print(f"reference shift {shift:+.2f} deg, wavelet scale {scale:.4f}, kappa {kappa}")
    for key, res in out["sets"].items():
        print("==", key)
        for name, r in res.items():
            al = " ".join(f"{l[:2]}:{'Y' if v['resolved'] else '.'}" for l, v in r["rows"].items() if "_z" in l)
            off = "".join("Y" if v["resolved"] else "." for l, v in r["rows"].items() if "offset" in l)
            print(f"  {name:24s} aligned {al}  min={r['min_resolvable_aligned_deg']}  offsets(P1..P4 x 5-20mm) {off}")
    print("kappa sensitivity (aligned P1..P4 | offsets):")
    for k, ok in sens.items():
        print(f"  {k:5s} " + "".join("Y" if ok[l] else "." for l in ok if "_z" in l) + " | "
              + "".join("Y" if ok[l] else "." for l in ok if "offset" in l))
    ad = out["axial_decay"]
    print("axial decay mean measured", np.round(ad["mean_measured"], 3), "predicted", np.round(ad["mean_predicted"], 3),
          f"rms {ad['rms_difference']:.3f}")


if __name__ == "__main__":
    main()
