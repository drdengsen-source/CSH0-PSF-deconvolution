# CSH0 C-scan deconvolution with transducer-derived PSFs

Code and processed data for the article

> S. Deng, Y. Yang, B. Zhao, G. Xing, F. Qian, L. He, P. Yang, *Deconvolution of CSH0 Guided-Wave Pipe C-Scans with Point-Spread Functions Derived from an Analytical PPM-EMAT Model* (submitted).

The repository contains what is needed to reproduce every number, table and data figure of the article: the analytical PPM-EMAT model, the two point-spread functions (PSFs) derived from it, the deconvolution code, and the processed measurement and simulation data on which the results are based.

## Quick start

```text
python -m pip install -r requirements.txt
python code/reproduce.py          # Sections 3-4: PSF validation, Tables 7-8, four-method comparison
python code/reproduce_pairs.py    # slot pairs P1-P4: Fig. 16, Table 9, measured axial decay, coherent restoration
python code/make_figures.py       # Figs. 12, 13, 15, 16 and 17 as PNG files in figures/
```

Results are written to `outputs/results.json` and `outputs/pairs.json`.

## Contents

| File | Content |
|---|---|
| `code/csh0_model.py` | Directivity $D(\alpha)$, axial PSF $h_z = D^2(\arctan(\Delta z/s))$ and circumferential PSF $h_\theta$ (Eqs. 3.3–3.4) |
| `code/deconvolution.py` | Non-negative L1 circular deconvolution (FISTA) |
| `code/coherent.py` | A-scan processing (band-pass, angle mapping, 1° binning) and coherent L1 restoration with the model wavelet |
| `code/reproduce.py` | Fig. 12 validation numbers, Tables 7–8, four-method comparison (Fig. 17) |
| `code/reproduce_pairs.py` | Slot pairs P1–P4: separability for all methods (Fig. 16, Table 9), parameter sensitivity, measured axial decay (Fig. 12(b)), coherent restoration (Section 5) |
| `code/make_figures.py` | Data figures of the article |
| `data/experimental_cscan.npz` | Measured C-scan amplitude image (mV), 65 circumferential samples (72–136°, 1° step) × 69 axial positions (35–375 mm, 5 mm step) |
| `data/simulated_axial_profiles.npz` | Peak envelope of the defect echo versus axial scan position (0–100 mm, 5 mm step) from full-wave simulations, original and optimized apertures |
| `data/slot_pairs_processed.npz` | Slot pairs P1–P4 and the D2 reference: analytic signal (V) of the band-passed, 64-fold averaged pulse-echo A-scans within the analysis window 60–150°, at the aligned axial position of each pair and at axial offsets of 5–20 mm (21 records), with labels, separations, axial positions and nominal slot angles |

## Method summary

**Operating point.** Design frequency 340 kHz, CSH0 wavenumber $k_0=0.663547$ mm⁻¹ ($\lambda_0=9.47$ mm). The experimental excitation was a five-cycle Hann-windowed burst at the 340 kHz design frequency. Pipe: 6061 aluminium alloy, outer radius 75 mm, wall thickness 3 mm. Apertures $(r_c,R_c)$: original (2, 8) mm, optimized (1, 15) mm. Defects D1–D5: axial slots at 40, 120, 200, 280 and 360 mm, nominal angle 100°, 8 mm long along the axis, 0.5–2 mm wide circumferentially, 0.5–2 mm deep (Table 6).

**C-scan formation.** At each axial position, the band-pass-filtered (200–480 kHz) A-scan envelope is mapped to the circumferential angle by $\theta(t)=[c(t-\tau_0)+d_0]/(2R)$ with $c=3160$ m/s, $d_0=75$ mm and $R=75$ mm (Eq. 3.1); the pixel value is the envelope maximum in each 1° bin. The circumferential coordinate is therefore a range coordinate.

**Circumferential PSF.** $h_\theta$ is the envelope of the pulse-echo wavelet: the burst spectrum multiplied by the PPM array response in transmission and in reception and by the band-pass filter (Eq. 3.3). It contains no fitted parameter. It is evaluated at the spectral peak of the received echoes (355.9 kHz); its −3 dB width is 9.81°, compared with 9.89° at 340 kHz.

**Deconvolution.** Each profile is multiplied by $\exp[G_B(\theta-\theta_p)]$ with $G_B=0.0131$ deg⁻¹ (Eq. 3.6), extended by a 20° mirror guard band at both ends and restored by non-negative L1 deconvolution with $\lambda=0.11\max\tilde I$ (Eq. 3.7, FISTA, ≤500 iterations).

**Slot pairs.** P1–P4 were machined on the same pipe after the C-scan, midway between the single slots and numbered from the D1 end: z = 80, 160, 240 and 320 mm; both slots of each pair are 8 × 1 × 2 mm (as D2), at 89°/111°, 92°/108°, 95°/105° and 97°/103° (separations 22°, 16°, 10° and 6°). A-scans are processed as the C-scan columns (200–480 kHz band-pass, Eq. 3.1 with τ0 = 6.25 µs, envelope maximum per 1° bin); this τ0 reproduces the C-scan column of D2 (correlation 0.996 over 86–136°). Angles are referenced to the D2 arrival (100°). Resolution criterion: both peaks within 2.5° of the slots and valley ≤ 81 % of the smaller peak (unchanged from Section 4.2).

**Coherent restoration (Section 5).** The analytic signal of the band-passed A-scan is modelled as a sparse sum of complex-weighted copies of the analytic model wavelet and restored by complex L1 (FISTA). The carrier of the model wavelet is matched to the carrier of the isolated D2 echo (357.7 kHz, time scale 0.950); λ = κ max|Aᴴz| with κ = 0.1, the smallest value for which the D2 reference restores as a single peak (secondary-peak level < 0.2) for all larger κ.

**SNR.** $20\log_{10}(p/\sigma_{bg})$, with $\sigma_{bg}$ pooled over the band 76–126° in the axial columns at least 20 mm from every defect.

**Axial PSF validation.** The simulated axial profile of each defect position is mapped to angle by $\alpha=\arctan(\Delta z/s)$, where $s=R(\theta_d-\theta_0)$ and the transducer sits at $\theta_0=28.65°$. In the original-aperture profile at 130°, the central sample (index 10) exceeds its neighbours by 36%, a single-sample numerical artefact; it is replaced by the mean of its two neighbours. The packaged array keeps the unmodified value.

## Scope

The repository contains processed data only.

## Licenses

Data and documentation: CC BY 4.0. Code in `code/`: MIT. See `LICENSE.md` and `code/LICENSE.txt`.
