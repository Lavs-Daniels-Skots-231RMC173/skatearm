"""QC camera pipeline (v1, classical CV — no learned models).

Measures the assembled unit from a PAIR of fixed cameras:

- overhead: peg presence + peg<->pocket ALIGNMENT in mm
- lateral:  exposed peg height -> INSERTION DEPTH estimate + coarse tilt

Segmentation is color-threshold based (flat-shaded sim parts):
  peg top   ~ (255, 245, 80)  -> "yellow" mask
  block top ~ (104, 241, 255) -> "cyan" mask
  (the wooden table (247,194,138) defeats naive 'orange' thresholds — measured)

mm-per-px comes from camera geometry (fovy + standoff), no checkerboard needed
in sim; the real cell will calibrate the usual way.

TWO PAIRS, ONE PIPELINE. ``FIXTURE_PAIR`` is the gate this file shipped with:
it inspects the unit held up at the fixture pose, and every committed camera-QC
figure was measured through it, so it stays ``measure()``'s default and nothing
about it moves. ``STATION_PAIR`` inspects the assembly station on the table
instead — the same overhead camera, which sees both, plus a second lateral
camera aimed at the station. That is the "second camera pair, not a
re-calibration" the weld-free cell was documented as needing: a cell that
releases the unit onto the table cannot present it to the fixture, so the optics
go to the part. Only *which* cameras are read and *from how far* is per-pair;
masks, ROI, band derivation and thresholds are shared, so a verdict from one
pair means the same thing as a verdict from the other.
"""
from collections import namedtuple

import mujoco
import numpy as np

PEG_LEN_MM = 40.0
TOP_CAM_Z, TOP_FOVY = 0.60, 42.0
SIDE_CAM_X, SIDE_FOVY = 0.32, 38.0
# The station's lateral camera is deliberately THE SAME LENS AT THE SAME
# STANDOFF as qc_side, mounted along -y instead of +x. Not a convenience: equal
# standoff and equal fovy give the two lateral views the same mm-per-px, so
# their depth estimates are directly comparable instead of merely both being
# "a side view". (-y rather than +x because the reject bin blocks the +x
# approach at the station — measured: 0 stub px, 0 rim px from there.)
STATION_CAM_Y, STATION_FOVY = SIDE_CAM_X, SIDE_FOVY

CameraPair = namedtuple(
    "CameraPair", "name top top_cam_z top_fovy side side_standoff side_fovy")

#: the shipped gate — the unit is carried to the cameras at the fixture pose
FIXTURE_PAIR = CameraPair("fixture", "qc_top", TOP_CAM_Z, TOP_FOVY,
                          "qc_side", SIDE_CAM_X, SIDE_FOVY)
#: in-situ optics aimed at the assembly station, for cells that cannot present
#: the unit to the fixture at all
STATION_PAIR = CameraPair("station", "qc_top", TOP_CAM_Z, TOP_FOVY,
                          "qc_station_side", STATION_CAM_Y, STATION_FOVY)


def _yellow(img):
    # peg top ~(255,245,80); the LIT TABLE is (247,194,138) and must stay out
    return (img[:, :, 0] > 220) & (img[:, :, 1] > 215) & (img[:, :, 2] < 120)


def _cyan(img):
    return (img[:, :, 0] < 140) & (img[:, :, 1] > 170) & (img[:, :, 2] > 200)


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    return float(xs.mean()), float(ys.mean())


def mm_per_px(standoff_m, fovy_deg, img_h):
    span = 2.0 * standoff_m * np.tan(np.radians(fovy_deg / 2))
    return span * 1000.0 / img_h


def _peg_colors(img):
    """Peg = bright-lit yellow top OR orange flank."""
    r, g, b = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    yellow = (r > 200) & (g > 190) & (b < 160)
    orange = (r > 190) & (g > 110) & (g < 195) & (b < 110)
    return yellow | orange


def stub_band_px(mm_per_px_side):
    """How many rows above the block's top edge may hold exposed peg.

    An exposed stub cannot be taller than the peg, so the search window is the
    peg's own length in pixels. This was the constant 70, and that constant was
    a silent ceiling: 70 px is 21.4 mm of peg, so the lowest depth the pipeline
    could ever report was PEG_LEN_MM - 21.4 - 2.0 = 16.58 mm — while
    ``verdict()``'s depth_min is 15.0. A 1.58 mm DEAD BAND, in which no unit,
    however badly seated, could drive depth_mm_est below the threshold that
    exists to reject it: walking the peg out of the bore, the estimate stopped
    at 16.58 mm while the oracle kept falling to 14.11 mm. Deriving the window
    from PEG_LEN_MM removes the ceiling instead of relocating it. Measured on
    both pairs, both cell paths, and at the adversarial pose where a gripper
    hangs directly over the unit: every seated reading is identical to the
    70-row one (stub px, rim px and depth alike), so the wider window changes
    nothing except the defects the old one clipped.
    """
    return int(np.ceil(PEG_LEN_MM / mm_per_px_side))


def measure(renderer, d, unit_z=0.135, roi_half=150, pair=FIXTURE_PAIR):
    """Render one camera PAIR and return measurements.

    Industrial-style FIXED INSPECTION WINDOW: the unit is always presented at a
    pose the pair is aimed at (image center), so analysis is restricted to a
    centered ROI — specular highlights on the arms and the wooden table outside
    the window cannot poison the segmentation (they did: a glint on the wrist
    dragged the 'peg centroid' 148 mm away in the first attempt).

    ``pair`` defaults to FIXTURE_PAIR, so every existing caller and every
    committed figure is unaffected."""
    out = {}
    renderer.update_scene(d, camera=pair.top)
    top = renderer.render().copy()
    renderer.update_scene(d, camera=pair.side)
    side = renderer.render().copy()
    h, w = top.shape[:2]
    cx, cy = w // 2, h // 2
    roi = np.zeros((h, w), bool)
    roi[cy - roi_half:cy + roi_half, cx - roi_half:cx + roi_half] = True

    # --- top view: presence + alignment (within ROI) ---
    ytop = _yellow(top) & roi
    ctop = _cyan(top) & roi
    peg_seed = _centroid(ytop)
    out["peg_present"] = bool(ytop.sum() > 150)
    # two-pass peg centroid: the strict yellow mask only catches the LIT side
    # of the peg top (centroid biased ~8 mm toward the light — measured).
    # Refine with a broad orange+yellow mask restricted to the seed's
    # neighborhood (table can't intrude there).
    peg_c = peg_seed
    if peg_seed:
        yy0, xx0 = np.mgrid[0:h, 0:w]
        near = (xx0 - peg_seed[0]) ** 2 + (yy0 - peg_seed[1]) ** 2 < 30 ** 2
        broad = (top[:, :, 0].astype(int) > 180) & (top[:, :, 2].astype(int) < 140) & near
        c2 = _centroid(broad)
        if c2:
            peg_c = c2
    mpp_t = mm_per_px(pair.top_cam_z - (unit_z + 0.025), pair.top_fovy, h)
    blk_c = None
    if peg_c:
        # ALIGNMENT REFERENCE = the POCKET RIM, not the whole block: the wrist
        # partially occludes one block corner and biased the full-block
        # centroid by ~7 mm (measured). The cyan rim ring within ~45 px of the
        # peg is what the peg must be concentric with anyway.
        yy, xx = np.mgrid[0:h, 0:w]
        ring = ctop & ((xx - peg_c[0]) ** 2 + (yy - peg_c[1]) ** 2 < 45 ** 2)
        blk_c = _centroid(ring)
    if peg_c and blk_c:
        out["align_err_mm"] = float(np.hypot(peg_c[0] - blk_c[0], peg_c[1] - blk_c[1]) * mpp_t)
    else:
        out["align_err_mm"] = None

    # --- side view: block first, then peg ONLY in the window above its top
    # edge (the orange wrist link and the table camouflage defeat any global
    # search — measured) ---
    cs_ = _cyan(side) & roi
    mpp_s = mm_per_px(pair.side_standoff, pair.side_fovy, h)
    band_px = stub_band_px(mpp_s)
    ys_ = np.zeros_like(cs_)
    if cs_.sum() > 100:
        blk_rows = np.nonzero(cs_.any(axis=1))[0]
        blk_cols = np.nonzero(cs_.any(axis=0))[0]
        top_row = blk_rows.min()
        band = np.zeros_like(cs_)
        band[max(0, top_row - band_px):top_row + 2, blk_cols.min():blk_cols.max() + 1] = True
        ys_ = _peg_colors(side) & band
        if ys_.sum() > 25:
            peg_rows = np.nonzero(ys_.any(axis=1))[0]
            exposed_px = max(0, top_row - peg_rows.min())
            exposed_mm = exposed_px * mpp_s
            out["depth_mm_est"] = float(PEG_LEN_MM - exposed_mm - 2.0)  # 2 mm floor-penetration bias
        else:
            out["depth_mm_est"] = None
    else:
        out["depth_mm_est"] = None
    # tilt: the 18 mm exposed stub is too small for a reliable axis estimate at
    # this resolution — explicitly v2 (higher-res camera or fitted edges);
    # the pose oracle keeps covering tilt and this is documented.
    out["tilt_deg_est"] = None

    out["_imgs"] = {"top": top, "side": side}
    out["_masks"] = {"top_peg": ytop, "top_blk": ctop, "side_peg": ys_, "side_blk": cs_}
    out["_centroids"] = {"peg": peg_c, "blk": blk_c}
    out["_mpp"] = {"top": mpp_t, "side": mpp_s}
    out["_pair"] = pair
    out["_band_px"] = band_px
    return out


def verdict(meas, depth_min=15.0, align_max=6.0, tilt_max=8.0):
    # tilt_deg_est may be None in v1 (covered by the oracle cross-check)
    ok = (meas.get("peg_present")
          and meas.get("align_err_mm") is not None and meas["align_err_mm"] <= align_max
          and meas.get("depth_mm_est") is not None and meas["depth_mm_est"] >= depth_min)
    if meas.get("tilt_deg_est") is not None:
        ok = ok and meas["tilt_deg_est"] <= tilt_max
    return "ACCEPT" if ok else "REJECT"


def annotate(meas, path_prefix):
    """Save annotated QC images (centroids, alignment line, values)."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    # top
    im = Image.fromarray(meas["_imgs"]["top"])
    dr = ImageDraw.Draw(im, "RGBA")
    pc, bc = meas["_centroids"]["peg"], meas["_centroids"]["blk"]
    if pc and bc:
        dr.line([pc, bc], fill=(255, 0, 0, 255), width=2)
        for c, col in ((pc, (255, 255, 0)), (bc, (0, 90, 255))):
            dr.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], outline=col, width=2)
    dr.rectangle([6, 6, 360, 52], fill=(10, 14, 24, 190))
    dr.text((12, 9), f"QC TOP  align err: {meas['align_err_mm']:.1f} mm", font=font, fill=(120, 220, 255))
    dr.text((12, 29), f"peg present: {meas['peg_present']}", font=font, fill=(230, 230, 230))
    im.save(path_prefix + "_top.png")
    # side
    im2 = Image.fromarray(meas["_imgs"]["side"])
    dr2 = ImageDraw.Draw(im2, "RGBA")
    dr2.rectangle([6, 6, 400, 52], fill=(10, 14, 24, 190))
    dd = meas["depth_mm_est"]
    tt = meas["tilt_deg_est"]
    dr2.text((12, 9), f"QC SIDE  depth est: {dd:.1f} mm" if dd else "QC SIDE  depth: n/a",
             font=font, fill=(120, 220, 255))
    dr2.text((12, 29), f"tilt est: {tt:.1f} deg" if tt is not None else "tilt: n/a",
             font=font, fill=(230, 230, 230))
    im2.save(path_prefix + "_side.png")
