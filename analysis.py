"""Exploratory Iba1 soma candidates. All distances are original-image pixels."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.morphology import disk, h_maxima
from skimage.segmentation import watershed

DEFAULTS = dict(green_sensitivity=65, background_sigma=15, background_strength=1,
                min_soma_area=18, max_soma_area=1800, core_radius=3,
                split_prominence=1.5, merge_saddle=0.75, dapi_sensitivity=55,
                min_nucleus_area=5, association_distance=6)


def settings(options):
    o = {k: float(options.get(k, v)) for k, v in DEFAULTS.items()}
    limits = dict(green_sensitivity=(1, 100), background_sigma=(2, 200),
                  background_strength=(0, 1), min_soma_area=(1, 100000),
                  max_soma_area=(1, 1000000), core_radius=(1, 20),
                  split_prominence=(0.2, 20), merge_saddle=(0.3, 0.95),
                  dapi_sensitivity=(1, 100), min_nucleus_area=(1, 100000),
                  association_distance=(0, 100))
    for k, v in o.items():
        if not np.isfinite(v) or not limits[k][0] <= v <= limits[k][1]:
            raise ValueError('Invalid setting: ' + k)
    if o['min_soma_area'] > o['max_soma_area']:
        raise ValueError('Minimum soma area must be <= maximum soma area.')
    return o


def normalize(a):
    """Float processing, no 8-bit quantization; record scale for reproducibility."""
    a = np.asarray(a)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError('Channels must contain finite nonnegative intensities.')
    scale = max(float(np.percentile(a, 99.9)), float(a.max()) * 0.01, 1e-12)
    return a.astype(np.float32) / scale, scale


def correct(a, sigma, strength):
    # Broad Gaussian estimate is fast and smooth, but can subtract broad somata.
    bg = ndi.gaussian_filter(a, sigma, mode='reflect')
    return np.maximum(a - strength * bg, 0), bg


def threshold(a, sensitivity):
    smooth = ndi.gaussian_filter(a, 0.7)
    lower = smooth[smooth <= np.percentile(smooth, 60)]
    med = float(np.median(lower))
    mad = float(np.median(np.abs(lower - med))) * 1.4826
    t = max(med + (7 - 0.055 * sensitivity) * mad,
            (0.35 - 0.0025 * sensitivity) * max(float(np.percentile(smooth, 99.5)), 1e-6))
    return smooth > t, t


def separate(mask, prominence, merge_saddle=None):
    """Split at thickness peaks; merge basins lacking a substantial neck."""
    d = ndi.distance_transform_edt(mask)
    if not mask.any():
        return np.zeros(mask.shape, np.int32)
    seeds = h_maxima(d, prominence) & mask
    # Always give each connected component a seed, including clipped border bodies.
    components, _ = ndi.label(mask)
    for i, box in enumerate(ndi.find_objects(components), 1):
        region = components[box] == i
        if not np.any(seeds[box] & region):
            local = np.where(region, d[box], -1)
            yy, xx = np.unravel_index(np.argmax(local), local.shape)
            seeds[yy + box[0].start, xx + box[1].start] = True
    markers, _ = ndi.label(seeds)
    labels = watershed(-d, markers, mask=mask).astype(np.int32)
    if merge_saddle is None:
        return labels
    peaks = ndi.maximum(d, labels, np.arange(labels.max() + 1))
    parent = list(range(labels.max() + 1))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    # Shared boundary thickness measures whether two seeds belong to one broad soma.
    edges = {}
    for axis in (0, 1):
        a = labels[:-1, :] if axis == 0 else labels[:, :-1]
        b = labels[1:, :] if axis == 0 else labels[:, 1:]
        da = d[:-1, :] if axis == 0 else d[:, :-1]
        db = d[1:, :] if axis == 0 else d[:, 1:]
        valid = (a > 0) & (b > 0) & (a != b)
        for i, j, saddle in zip(a[valid], b[valid], np.minimum(da, db)[valid]):
            key = tuple(sorted((int(i), int(j))))
            edges[key] = max(edges.get(key, 0), float(saddle))
    for (i, j), saddle in sorted(edges.items(), key=lambda x: -x[1]):
        if saddle / max(min(peaks[i], peaks[j]), 1e-6) >= merge_saddle:
            parent[root(j)] = root(i)
    lut = np.array([root(i) for i in range(len(parent))])
    merged = lut[labels]
    ids = np.unique(merged[merged > 0])
    new = np.zeros(len(parent), np.int32)
    new[ids] = np.arange(1, len(ids) + 1)
    return new[merged]


def count_candidates(rgb, options=None):
    options = options or {}
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('Select a single field with green and blue channels.')
    o = settings(options)
    h, w = a.shape[:2]
    x0, y0, x1, y1 = [int(options.get(k, v)) for k, v in zip(
        ('x0', 'y0', 'x1', 'y1'), (0, 0, w, h))]
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError('ROI must lie inside the image.')
    # Correct full field before cropping to avoid artificial background at ROI edges.
    g, gs = normalize(a[:, :, 1]); b, bs = normalize(a[:, :, 2])
    gc, bg = correct(g, o['background_sigma'], o['background_strength'])
    bc, _ = correct(b, o['background_sigma'], o['background_strength'])
    crop = (slice(y0, y1), slice(x0, x1))
    support, gt = threshold(gc[crop], o['green_sensitivity'])
    nuclear, bt = threshold(bc[crop], o['dapi_sensitivity'])
    radius = int(o['core_radius'])
    padded = np.pad(support, radius * 2, mode='edge')
    opened = ndi.binary_opening(padded, structure=disk(radius))
    soma = opened[radius*2:-radius*2, radius*2:-radius*2]
    # Holes represent possible nuclei; filling does not itself supply green evidence.
    soma = ndi.binary_fill_holes(soma)
    labels = separate(soma, o['split_prominence'], o['merge_saddle'])
    nuclear = ndi.binary_fill_holes(nuclear)
    nuclei = separate(nuclear, 1.0)
    sizes = np.bincount(nuclei.ravel())
    kept = np.flatnonzero(sizes >= o['min_nucleus_area']); kept = kept[kept > 0]
    nc = ndi.center_of_mass(nuclear, nuclei, kept) if len(kept) else []
    cells = []; noise = 0
    for i, box in enumerate(ndi.find_objects(labels), 1):
        if box is None: continue
        region = labels[box] == i
        area = int(region.sum())
        # Report tiny regions; preserve borderline-size candidates for review.
        if area < max(3, o['min_soma_area'] / 3):
            noise += 1; labels[box][region] = 0; continue
        yy, xx = np.nonzero(region)
        weights = gc[crop][box][region] + 1e-6
        cy, cx = float(np.average(yy, weights=weights)), float(np.average(xx, weights=weights))
        nearest = np.argmin((yy-cy)**2 + (xx-cx)**2)
        cy, cx = float(yy[nearest]+box[0].start), float(xx[nearest]+box[1].start)
        border = box[0].start == 0 or box[1].start == 0 or box[0].stop == y1-y0 or box[1].stop == x1-x0
        reasons = []
        if area < o['min_soma_area']: reasons.append('small_soma')
        if area > o['max_soma_area']: reasons.append('large_or_merged_soma')
        if max(region.shape) / max(min(region.shape), 1) > 4: reasons.append('elongated_region')
        # Nucleus-centre distance to the soma footprint, not overlap fraction gating.
        tree = cKDTree(np.column_stack((yy+box[0].start, xx+box[1].start)))
        distances = tree.query(np.asarray(nc).reshape(-1, 2))[0] if len(nc) else []
        assoc = [(int(nid), float(distance)) for nid, distance in zip(kept, distances)]
        assoc = sorted((v for v in assoc if v[1] <= o['association_distance']), key=lambda v: v[1])
        if not assoc: reasons.append('no_nearby_DAPI')
        if len(assoc) > 1: reasons.append('multiple_nearby_nuclei')
        support_fraction = float((region & support[box]).sum()/area)
        if support_fraction < .45: reasons.append('weak_green_support')
        edge = region & ~ndi.binary_erosion(region)
        ey, ex = np.nonzero(edge)
        cells.append(dict(id=len(cells)+1, x=cx+x0, y=cy+y0,
            original_x=cx+x0, original_y=cy+y0, source='automatic',
            review_status='unreviewed', border=bool(border), reasons=reasons,
            soma_area_px=area, green_support_fraction=support_fraction,
            nucleus_ids=[v[0] for v in assoc], nucleus_distance_px=assoc[0][1] if assoc else None,
            boundary_px=np.column_stack((ex+box[1].start+x0, ey+box[0].start+y0)).tolist()))
    owners = {}
    for c in cells:
        for n in c['nucleus_ids']: owners.setdefault(n, []).append(c)
    for group in owners.values():
        if len(group) > 1:
            for c in group:
                if 'shared_DAPI_review_split' not in c['reasons']: c['reasons'].append('shared_DAPI_review_split')
    for c in cells: c['uncertain'] = bool(c['reasons'])
    full_labels = np.zeros((h, w), np.int32); full_labels[crop] = labels
    full_support = np.zeros((h, w), bool); full_support[crop] = support
    o.update(x0=x0, y0=y0, x1=x1, y1=y1, green_scale=gs, blue_scale=bs,
             corrected_green_threshold=gt, corrected_blue_threshold=bt)
    return dict(cells=cells, settings=o, noise_regions=noise,
                method_version='1.0-soma-review',
                arrays=dict(corrected=gc, background=bg, soma=full_labels,
                            processes=full_support & (full_labels == 0)))


def evaluate_points(cells, reference, tolerance=8):
    """One-to-one point comparison; meaningful only for a fully annotated field/ROI."""
    pred = np.array([[c['x'], c['y']] for c in cells], float).reshape(-1, 2)
    ref = np.array([[c['x'], c['y']] for c in reference], float).reshape(-1, 2)
    matched = []; duplicate_like = 0
    if len(pred) and len(ref):
        distances = np.linalg.norm(pred[:, None] - ref[None, :], axis=2)
        cost = np.where(distances <= tolerance, distances, 1e9)
        rows, cols = linear_sum_assignment(cost)
        matched = [(int(i), int(j)) for i, j in zip(rows, cols) if distances[i, j] <= tolerance]
        used = {i for i, j in matched}; covered = {j for i, j in matched}
        duplicate_like = sum(any(distances[i, j] <= tolerance for j in covered)
                             for i in range(len(pred)) if i not in used)
    tp = len(matched); fp = len(pred)-tp; fn = len(ref)-tp
    return dict(true_positives=tp, false_positives=fp, missed_cells=fn,
                duplicate_like_extras=int(duplicate_like),
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                tolerance_px=tolerance, matched_pairs=matched)
