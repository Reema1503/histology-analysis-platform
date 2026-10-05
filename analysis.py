"""Transparent baseline for Iba1/DAPI images; not a trained AI model."""
import numpy as np
from scipy import ndimage as ndi


def otsu(a):
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    p = hist.astype(float) / max(1, hist.sum())
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    score = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return float(np.argmax(score[:-1]))


def count_candidates(rgb, options):
    """DAPI components with above-threshold green signal in a nuclear surround.

    Touching nuclei can remain merged. The ring is a screening proxy for Iba1
    association, not evidence that the signal belongs to that nucleus.
    """
    h, w = rgb.shape[:2]
    x0, y0 = int(options.get('x0', 0)), int(options.get('y0', 0))
    x1, y1 = int(options.get('x1', w)), int(options.get('y1', h))
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI must be inside the image and have positive width and height.')
    crop = rgb[y0:y1, x0:x1]
    blue = crop[:, :, 2].astype(float)
    green = crop[:, :, 1].astype(float)
    red = crop[:, :, 0].astype(float)
    blue_smooth = ndi.gaussian_filter(blue, 1)
    # Identify blue-dominant nuclear pixels, not just any signal in RGB blue.
    # Ratios are adjustable because bright overlapping Iba1 may affect RGB hues.
    dominance = float(options.get('blue_dominance', 1.2))
    min_contrast = float(options.get('min_blue_contrast', 15))
    background = ndi.median_filter(blue, size=21)
    dapi_support = ((blue > dominance * np.maximum(green, red)) &
                    (blue - background >= min_contrast))
    auto = otsu(blue_smooth)
    bt = auto if options.get('auto_blue', True) else float(options['blue_threshold'])
    gt = float(options.get('green_threshold', 30))
    minimum = int(options.get('min_area', 20))
    maximum = int(options.get('max_area', 1500))
    radius = int(options.get('radius', 5))
    fraction = float(options.get('positive_fraction', 0.15))
    if not (0 <= bt <= 255 and 0 <= gt <= 255 and 1 <= minimum <= maximum and 1 <= radius <= 30 and 0 <= fraction <= 1 and 1 <= dominance <= 3 and 0 <= min_contrast <= 255):
        raise ValueError('Invalid detection parameters.')
    mask = ndi.binary_fill_holes((blue_smooth > bt) & dapi_support)
    labels, _ = ndi.label(mask)
    objects = ndi.find_objects(labels)
    cells = []
    for label_id, sl in enumerate(objects, 1):
        if sl is None:
            continue
        area = int(np.count_nonzero(labels[sl] == label_id))
        if not minimum <= area <= maximum:
            continue
        nuclear_pixels = labels[sl] == label_id
        supported = nuclear_pixels & dapi_support[sl]
        if np.count_nonzero(supported) < minimum:
            continue
        sy = slice(max(0, sl[0].start-radius), min(crop.shape[0], sl[0].stop+radius))
        sx = slice(max(0, sl[1].start-radius), min(crop.shape[1], sl[1].stop+radius))
        nucleus = labels[sy, sx] == label_id
        ring = ndi.binary_dilation(nucleus, iterations=radius) & (labels[sy, sx] == 0)
        if not ring.any():
            continue
        signal = green[sy, sx][ring]
        positive = float(np.mean(signal > gt))
        if positive < fraction:
            continue
        yy, xx = np.nonzero(labels[sl] == label_id)
        # Anchor the overlay to an actual supported DAPI pixel nearest the centroid.
        syy, sxx = np.nonzero(supported)
        nearest = np.argmin((sxx-xx.mean())**2 + (syy-yy.mean())**2)
        cells.append({'id': len(cells)+1, 'x': float(sxx[nearest]+sl[1].start+x0),
                      'y': float(syy[nearest]+sl[0].start+y0), 'nuclear_area_px': area,
                      'mean_blue_nucleus': float(blue[sl][supported].mean()),
                      'mean_blue_contrast': float((blue[sl]-background[sl])[supported].mean()),
                      'mean_green_ring': float(signal.mean()), 'positive_fraction': positive,
                      'source': 'automatic_candidate'})
    return {'cells': cells, 'blue_threshold': bt, 'roi_area_px': int(crop.shape[0]*crop.shape[1]),
            'method': 'Blue-dominant, locally contrasted DAPI components with Iba1-positive nuclear surrounds',
            'method_version': '0.2'}
