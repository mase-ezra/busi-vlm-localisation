'''
Used similar approach to BUSClean for caliper detection and ultrasound annotation removal as well as EasyOCR for text detection. 
See: https://github.com/hawaii-ai/bus-cleaning
See: https://github.com/JaidedAI/EasyOCR
'''

import cv2
import numpy as np
from itertools import combinations
import easyocr
import torch

ocr_reader = easyocr.Reader(['en'], gpu=True)

# Tunable parameters.

# CC-based caliper detection.
CALIPER_BINARY_THRESH = 200
CALIPER_MIN_AREA = 30
CALIPER_MAX_AREA = 500
CALIPER_MIN_SIDE = 8
CALIPER_MAX_SIDE = 40
CALIPER_MIN_ASPECT = 0.7
CALIPER_MAX_ASPECT = 1.3
CALIPER_MIN_FILL = 0.15
CALIPER_MAX_FILL = 0.35

# Template matching fallback (runs if cc finds < MIN_CALIPERS_BEFORE_TEMPLATE).
TEMPLATE_BRIGHT_THRESH = 150
TEMPLATE_ARM_LENGTHS = (4, 5, 6, 7, 8)
TEMPLATE_THICKNESS = 2
TEMPLATE_MATCH_THRESHOLD = 0.70
TEMPLATE_NMS_RADIUS = 10
CC_TEMPLATE_DEDUP_RADIUS = 12
MIN_CALIPERS_BEFORE_TEMPLATE = 4

# Annotation colour sampling.
CALIPER_BINARY_THRESH_SAMPLE = 200
COLOUR_TOLERANCE = 8
COLOUR_FLOOR = 150

# Adaptive coverage clamp (for line scoring only, masking uses raw range).
MASK_COVERAGE_LIMIT = 0.50
MASK_COVERAGE_TARGET = 0.15
MASK_TIGHTEN_MAX_ITER = 50

# Paired dotted line scoring.
LINE_MIN_DOTNESS = 0.1
LINE_MIN_LENGTH_PX = 20
LINE_HALFWIDTH = 5
LINE_SAMPLE_HALFWIDTH = 2
LINE_SKIP_FRACTION = 0.08

# Radial trace fallback for unpaired calipers.
TRACE_RING_INNER = 18
TRACE_SLOT_LENGTH = 80
TRACE_SLOT_HALFWIDTH = 3
TRACE_N_ANGLES = 360
TRACE_SMOOTH_KERNEL_SIZE = 35
TRACE_MIN_DIRECTION_SCORE = 4
TRACE_STEP_PX = 4
TRACE_PATIENCE_STEPS = 15
TRACE_MAX_STEPS = 500
TRACE_HALFWIDTH = 5

# Digit label padding.
CALIPER_DIGIT_PADDING = 12

# OCR text detection.
OCR_CONF = 0.05  # 5% gets cut-off text without false positives.
OCR_PADDING = 10
OCR_IGNORE_STRINGS = {'4', '2+', '2'}
TEXT_BOX_MAX_AREA = 10000

# Coloured annotation mask (non-greyscale).
COLOUR_DIFF_THRESH = 12
NON_GREY_MIN_BRIGHT = 20  # Helps with anti-aliasing artifacts.

# Inpainting.
INPAINT_RADIUS = 3
MASK_DILATE_PX = 2

#Create thin rectangle (around dotted line)
def build_oriented_rectangle(p1, p2, halfwidth):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-6:
        hw = halfwidth
        return np.array([[x1-hw, y1-hw], [x1+hw, y1-hw],
                         [x1+hw, y1+hw], [x1-hw, y1+hw]], dtype=np.int32)
    px = (-dy / length) * halfwidth
    py = (dx / length) * halfwidth
    return np.array([[x1+px, y1+py], [x2+px, y2+py],
                     [x2-px, y2-py], [x1-px, y1-py]], dtype=np.int32)


#combine boxes that have touching edges, repeat until completed
def merge_overlapping_boxes(boxes, dilate=3):
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, x2, y2 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                ax1, ay1, ax2, ay2 = boxes[j]
                if not (x2+dilate < ax1 or ax2+dilate < x1 or
                        y2+dilate < ay1 or ay2+dilate < y1):
                    x1, y1 = min(x1, ax1), min(y1, ay1)
                    x2, y2 = max(x2, ax2), max(y2, ay2)
                    used[j] = True
                    changed = True
            merged.append([x1, y1, x2, y2])
            used[i] = True
        boxes = merged
    return [tuple(b) for b in boxes]

# CC-based caliper detection.
def find_caliper_markers_cc(gray):
    _, binary = cv2.threshold(gray, CALIPER_BINARY_THRESH, 255, cv2.THRESH_BINARY)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    raw_boxes = []
    for i in range(1, n_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if not (CALIPER_MIN_AREA <= area <= CALIPER_MAX_AREA): continue
        if not (CALIPER_MIN_SIDE <= w <= CALIPER_MAX_SIDE): continue
        if not (CALIPER_MIN_SIDE <= h <= CALIPER_MAX_SIDE): continue
        aspect = w / max(h, 1)
        if not (CALIPER_MIN_ASPECT <= aspect <= CALIPER_MAX_ASPECT): continue
        fill = area / (w * h + 1e-6)
        if not (CALIPER_MIN_FILL <= fill <= CALIPER_MAX_FILL): continue

        raw_boxes.append((x, y, x+w, y+h))

    return merge_overlapping_boxes(raw_boxes, dilate=4)

# Template-matching caliper detection (fallback).
def _make_plus_template(arm_length, thickness):
    size = 2 * arm_length + 1
    tmpl = np.zeros((size, size), dtype=np.float32)
    c, ht = arm_length, thickness // 2
    tmpl[:, c-ht:c+ht+1] = 1.0
    tmpl[c-ht:c+ht+1, :] = 1.0
    return tmpl

def find_caliper_markers_template(gray):
    bw_raw = (gray >= TEMPLATE_BRIGHT_THRESH).astype(np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    bw = cv2.morphologyEx(bw_raw, cv2.MORPH_CLOSE, kernel).astype(np.float32)

    all_matches = []
    for arm in TEMPLATE_ARM_LENGTHS:
        tmpl = _make_plus_template(arm, TEMPLATE_THICKNESS)
        response = cv2.matchTemplate(bw, tmpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(response >= TEMPLATE_MATCH_THRESHOLD)
        for y, x in zip(ys.tolist(), xs.tolist()):
            all_matches.append((x+arm, y+arm, arm, float(response[y, x])))

    if not all_matches:
        return []

    all_matches.sort(key=lambda m: -m[3])
    kept = []
    for cx, cy, arm, score in all_matches:
        if not any(abs(cx-kx) <= TEMPLATE_NMS_RADIUS and
                   abs(cy-ky) <= TEMPLATE_NMS_RADIUS
                   for kx, ky, *_ in kept):
            kept.append((cx, cy, arm, score))

    return [(cx-arm-2, cy-arm-2, cx+arm+2, cy+arm+2) for cx, cy, arm, _ in kept]

# Combined caliper detection, only run if not enough callipers are detected
def find_caliper_markers(gray):
    cc_boxes = find_caliper_markers_cc(gray)

    if len(cc_boxes) >= MIN_CALIPERS_BEFORE_TEMPLATE:
        return cc_boxes

    tm_boxes = find_caliper_markers_template(gray)
    cc_centres = [((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in cc_boxes]
    combined = list(cc_boxes)

    for tb in tm_boxes:
        tcx, tcy = (tb[0]+tb[2])//2, (tb[1]+tb[3])//2
        if not any(abs(tcx-ccx) <= CC_TEMPLATE_DEDUP_RADIUS and
                   abs(tcy-ccy) <= CC_TEMPLATE_DEDUP_RADIUS
                   for ccx, ccy in cc_centres):
            combined.append(tb)

    return merge_overlapping_boxes(combined, dilate=4)

# Sample annotation colour range, used for detecting colour band of callipers
def sample_annotation_colours(gray, caliper_boxes):
    if not caliper_boxes:
        return COLOUR_FLOOR, 255

    pixel_values = []
    for x1, y1, x2, y2 in caliper_boxes:
        roi = gray[y1:y2, x1:x2]
        bright = roi[roi > CALIPER_BINARY_THRESH_SAMPLE]
        if bright.size:
            pixel_values.extend(bright.tolist())

    if not pixel_values:
        return COLOUR_FLOOR, 255

    lo = max(COLOUR_FLOOR, int(min(pixel_values)) - COLOUR_TOLERANCE)
    hi = min(255, int(max(pixel_values)) + COLOUR_TOLERANCE)
    return lo, hi

#if colour band masks too much of the bounding box, shrink colour range until it meets threshold
def clamp_mask_coverage(gray, colour_lo, colour_hi):
    total = gray.size
    for _ in range(MASK_TIGHTEN_MAX_ITER):
        if np.sum((gray >= colour_lo) & (gray <= colour_hi)) / total <= MASK_COVERAGE_LIMIT:
            break
        if colour_lo + 1 >= colour_hi:
            break
        colour_lo += 1
        colour_hi -= 1
        if np.sum((gray >= colour_lo) & (gray <= colour_hi)) / total <= MASK_COVERAGE_TARGET:
            break
    return colour_lo, colour_hi

# Line scoring, go along line and check fraction of samples that land on annotation coloured pixels
def _score_line(gray, p1, p2, colour_lo, colour_hi):
    x1, y1 = p1
    x2, y2 = p2
    length_px = int(np.hypot(x2-x1, y2-y1))
    if length_px < LINE_MIN_LENGTH_PX:
        return 0.0, length_px

    img_h, img_w = gray.shape
    n_samples = max(length_px // 2, 6)
    hw, skip = LINE_SAMPLE_HALFWIDTH, LINE_SKIP_FRACTION
    hit = total = 0

    for t in np.linspace(skip, 1.0-skip, n_samples):
        cx = int(x1 + (x2-x1)*t)
        cy = int(y1 + (y2-y1)*t)
        patch = gray[max(0, cy-hw):min(img_h, cy+hw+1),
                     max(0, cx-hw):min(img_w, cx+hw+1)]
        if patch.size == 0:
            continue
        if colour_lo <= int(patch.max()) <= colour_hi:
            hit += 1
        total += 1

    return (hit / total if total else 0.0), length_px

# Pair calipers into dotted lines (with false-positive guard).
def find_dotted_lines(gray, caliper_boxes, colour_lo, colour_hi, img_shape):
    if len(caliper_boxes) < 2:
        return [], set()

    img_h, img_w = img_shape
    max_lines = len(caliper_boxes) // 2  # Hard cap
    centres = [((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in caliper_boxes]

    candidates = []
    for i, j in combinations(range(len(centres)), 2):
        score, length = _score_line(gray, centres[i], centres[j], colour_lo, colour_hi)
        if length >= LINE_MIN_LENGTH_PX and score >= LINE_MIN_DOTNESS:
            candidates.append((score, length, i, j))

    candidates.sort(reverse=True)
    results, used = [], set()

    for score, length, i, j in candidates:
        # Hard cap prevents false positives from tissue speckle.
        if len(results) >= max_lines:
            break

        if i in used or j in used:
            continue

        used.add(i)
        used.add(j)

        poly = build_oriented_rectangle(centres[i], centres[j], LINE_HALFWIDTH)
        poly[:, 0] = np.clip(poly[:, 0], 0, img_w-1)
        poly[:, 1] = np.clip(poly[:, 1], 0, img_h-1)

        results.append({
            'polygon': poly,
            'bbox': (int(poly[:, 0].min()), int(poly[:, 1].min()),
                     int(poly[:, 0].max()), int(poly[:, 1].max())),
            'score': float(score),
            'pair': (i, j),
        })

    return results, used

# Radial direction finder (vectorised). this is really dodgy and doesnt work too well honestly, idea was to sweep around the calliper 360 degrees, then find the direction that has
# has dotted line extending a minimum direction outwards. To be used if the image has an uneven amount of callipers
def _find_outward_direction(gray, start_xy, colour_lo, colour_hi):
    h, w = gray.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    bw = ((gray >= colour_lo) & (gray <= colour_hi)).astype(np.uint8)

    N = TRACE_N_ANGLES
    thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    angle_scores = np.zeros(N, dtype=np.float32)

    for r in range(TRACE_RING_INNER, TRACE_RING_INNER + TRACE_SLOT_LENGTH):
        for w_off in range(-TRACE_SLOT_HALFWIDTH, TRACE_SLOT_HALFWIDTH + 1):
            pxs = np.round(sx + r*cos_t - w_off*sin_t).astype(np.int32)
            pys = np.round(sy + r*sin_t + w_off*cos_t).astype(np.int32)
            valid = (pxs >= 0) & (pxs < w) & (pys >= 0) & (pys < h)
            safe_px = np.where(valid, pxs, 0)
            safe_py = np.where(valid, pys, 0)
            hits = valid & bw[safe_py, safe_px].astype(bool)
            angle_scores += hits.astype(np.float32)

    ks = TRACE_SMOOTH_KERNEL_SIZE
    padded = np.concatenate([angle_scores[-ks:], angle_scores, angle_scores[:ks]])
    smooth = np.convolve(padded, np.ones(ks)/ks, mode='same')[ks:ks+N]

    half = N // 2
    axis_scores = smooth[:half] + smooth[half:]

    if axis_scores.max() < TRACE_MIN_DIRECTION_SCORE:
        return None

    best_axis = int(np.argmax(axis_scores))
    score_a = float(smooth[best_axis])
    score_b = float(smooth[(best_axis + half) % N])
    theta = thetas[best_axis] if score_a >= score_b else thetas[(best_axis + half) % N]

    return float(np.cos(theta)), float(np.sin(theta)), max(score_a, score_b)

# trace the dotted line, moving outwards in the direction determined by find outward direction above, give up after too many consecutive misses
def _trace_dotted_line(gray, start_xy, colour_lo, colour_hi):
    direction = _find_outward_direction(gray, start_xy, colour_lo, colour_hi)
    if direction is None:
        return None

    dx, dy = direction[0], direction[1]
    h, w = gray.shape
    sx, sy = start_xy
    hw = LINE_SAMPLE_HALFWIDTH
    last_hit = None
    no_hit = 0
    hit_cnt = 0

    for step in range(TRACE_MAX_STEPS):
        r = TRACE_RING_INNER + step * TRACE_STEP_PX
        x = int(round(sx + r*dx))
        y = int(round(sy + r*dy))
        if not (0 <= x < w and 0 <= y < h):
            break
        patch = gray[max(0, y-hw):min(h, y+hw+1),
                     max(0, x-hw):min(w, x+hw+1)]
        if patch.size == 0:
            break
        if colour_lo <= int(patch.max()) <= colour_hi:
            last_hit = (x, y)
            no_hit = 0
            hit_cnt += 1
        else:
            no_hit += 1
            if no_hit >= TRACE_PATIENCE_STEPS:
                break

    if last_hit is None or hit_cnt < 3:
        return None
    if np.hypot(last_hit[0]-sx, last_hit[1]-sy) < TRACE_RING_INNER + 5:
        return None

    return (dx, dy), last_hit


# run radial trace on any calliper thand didnt get paired with another calliper
def find_traced_lines(gray, caliper_boxes, paired_indices,
                      colour_lo, colour_hi, img_shape):
    img_h, img_w = img_shape
    centres = [((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in caliper_boxes]
    results = []

    for i, centre in enumerate(centres):
        if i in paired_indices:
            continue
        trace = _trace_dotted_line(gray, centre, colour_lo, colour_hi)
        if trace is None:
            continue
        _, end_xy = trace
        poly = build_oriented_rectangle(centre, end_xy, TRACE_HALFWIDTH)
        poly[:, 0] = np.clip(poly[:, 0], 0, img_w-1)
        poly[:, 1] = np.clip(poly[:, 1], 0, img_h-1)
        results.append({
            'polygon': poly,
            'bbox': (int(poly[:, 0].min()), int(poly[:, 1].min()),
                     int(poly[:, 0].max()), int(poly[:, 1].max())),
            'score': 1.0,
            'pair': (i, None),
            'traced': True,
        })

    return results

# Expand caliper boxes / expand calliper boxes a bit to detect additional artifacts around the calliper, especially numbers 1, 2
def expand_caliper_boxes(caliper_boxes, img_h, img_w,
                         padding=CALIPER_DIGIT_PADDING):
    return [
        (max(0, x1-padding), max(0, y1-padding),
         min(img_w, x2+padding), min(img_h, y2+padding))
        for x1, y1, x2, y2 in caliper_boxes
    ]
# OCR detecting and masking

def find_text_regions_easyocr(img_bgr, reader):
    boxes = []

    for bbox, text, conf in reader.readtext(img_bgr):
        text_clean = text.strip().lower()

        if text_clean in OCR_IGNORE_STRINGS:
            continue

        if conf < OCR_CONF:
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        x1 = max(0, int(min(xs)) - OCR_PADDING)
        y1 = max(0, int(min(ys)) - OCR_PADDING)
        x2 = min(img_bgr.shape[1], int(max(xs)) + OCR_PADDING)
        y2 = min(img_bgr.shape[0], int(max(ys)) + OCR_PADDING)

        if TEXT_BOX_MAX_AREA > 0 and (x2 - x1) * (y2 - y1) > TEXT_BOX_MAX_AREA:
            continue

        boxes.append((x1, y1, x2, y2))

    return boxes

# Build pixel-exact removal mask.
def build_colour_exact_mask(gray, img_bgr, regions,
                             colour_lo, colour_hi,
                             dilate_px=MASK_DILATE_PX):
    """
    Builds three component masks and combines them:
    grey_mask - pixels in [colour_lo, colour_hi] inside annotation regions
    colour_mask - non-greyscale (coloured overlay) pixels
    text_mask - greyscale annotation pixels inside ocr text regions
    """
    h, w = gray.shape

    region_mask = np.zeros((h, w), dtype=np.uint8)
    text_region = np.zeros((h, w), dtype=np.uint8)

    for det in regions:
        if det.get('polygon') is not None:
            cv2.fillPoly(region_mask, [det['polygon'].astype(np.int32)], 255)
        else:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(region_mask, (x1, y1), (x2, y2), 255, -1)
        if det.get('kind') == 'text':
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(text_region, (x1, y1), (x2, y2), 255, -1)

    # Greyscale annotation mask.
    grey_pixel_mask = ((gray >= colour_lo) & (gray <= colour_hi)).astype(np.uint8) * 255
    grey_mask = cv2.bitwise_and(region_mask, grey_pixel_mask)

    # Separate text portion from non-text portion.
    text_mask = cv2.bitwise_and(grey_mask, text_region)
    grey_mask_no_text = cv2.bitwise_and(grey_mask, cv2.bitwise_not(text_region))

    # Coloured annotation mask.
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)

    max_diff = np.maximum.reduce([np.abs(r-g), np.abs(r-b), np.abs(g-b)])
    brightness = np.maximum.reduce([r, g, b])

    colour_flag = ((max_diff >= COLOUR_DIFF_THRESH) &
                   (brightness >= NON_GREY_MIN_BRIGHT)).astype(np.uint8) * 255

    colour_mask = colour_flag

    mask = cv2.bitwise_or(grey_mask_no_text, colour_mask)
    mask = cv2.bitwise_or(mask, text_mask)

    if dilate_px > 0:
        k = 2*dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    return mask, grey_mask_no_text, colour_mask, text_mask

#inpaint the mask
def inpaint_masked(img_bgr, mask, radius=INPAINT_RADIUS):
    return cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA)

#fallback for suspicous image, if there are too few callipers or lines found
def sus_img(gray, img_bgr, caliper_boxes_raw, dotted_lines):
    img_h, img_w = gray.shape

    # Add template-matched calipers if we are short.
    if len(caliper_boxes_raw) < MIN_CALIPERS_BEFORE_TEMPLATE:
        tm_boxes = find_caliper_markers_template(gray)

        cc_centres = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in caliper_boxes_raw]

        for tb in tm_boxes:
            tcx = (tb[0] + tb[2]) // 2
            tcy = (tb[1] + tb[3]) // 2

            if not any(abs(tcx - ccx) <= CC_TEMPLATE_DEDUP_RADIUS and abs(tcy - ccy) <= CC_TEMPLATE_DEDUP_RADIUS for ccx, ccy in cc_centres):
                caliper_boxes_raw.append(tb)

        caliper_boxes_raw = merge_overlapping_boxes(caliper_boxes_raw, dilate=4)

    # Compute both colour ranges.
    colour_lo_raw, colour_hi_raw = sample_annotation_colours(gray, caliper_boxes_raw)

    colour_lo, colour_hi = clamp_mask_coverage(gray, colour_lo_raw, colour_hi_raw)

    # Re-score dotted lines with clamped range.
    dotted_lines, paired_indices = find_dotted_lines(gray, caliper_boxes_raw, colour_lo, colour_hi, (img_h, img_w))

    # Radial trace using the raw range so dim dots are not missed.
    traced_lines = find_traced_lines(gray, caliper_boxes_raw, paired_indices, colour_lo_raw, colour_hi_raw, (img_h, img_w))

    all_lines = dotted_lines + traced_lines

    return (caliper_boxes_raw, all_lines, colour_lo, colour_hi, colour_lo_raw, colour_hi_raw)

#MAIN PIPELINE
def preprocess_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    # Detect caliper markers.
    caliper_boxes_raw = find_caliper_markers(gray)

    # Sample annotation colour range
    colour_lo_raw, colour_hi_raw = sample_annotation_colours(gray, caliper_boxes_raw)
    colour_lo, colour_hi = clamp_mask_coverage(gray, colour_lo_raw, colour_hi_raw)

    # Pair calipers into dotted lines.
    dotted_lines, paired_indices = find_dotted_lines(
        gray, caliper_boxes_raw, colour_lo, colour_hi, (img_h, img_w)
    )

    traced_lines = []

    # Fallback check.
    n_cal = len(caliper_boxes_raw)
    n_lines = len(dotted_lines)
    need_fallback = (
        (n_cal < 4 or n_lines < 2) and
        not (n_cal == 0 and n_lines == 0)
    )

    if need_fallback:
        caliper_boxes_raw, all_lines, colour_lo, colour_hi, colour_lo_raw, colour_hi_raw = sus_img(
            gray, img_bgr, caliper_boxes_raw, dotted_lines
        )
        dotted_lines = [l for l in all_lines if not l.get('traced')]
        traced_lines = [l for l in all_lines if l.get('traced')]
        paired_indices = set()
        for l in dotted_lines:
            i, j = l['pair']
            paired_indices.add(i)
            if j is not None:
                paired_indices.add(j)

    # OCR text detection + expand caliper boxes.
    text_boxes = find_text_regions_easyocr(img_bgr, ocr_reader)
    caliper_boxes = expand_caliper_boxes(caliper_boxes_raw, img_h, img_w)

    # Assemble regions and build mask.
    all_lines = dotted_lines + traced_lines
    all_regions = []

    for bbox in caliper_boxes:
        all_regions.append({'bbox': bbox, 'polygon': None, 'kind': 'caliper'})
    for line in all_lines:
        all_regions.append({'bbox': line['bbox'], 'polygon': line['polygon'],
                            'kind': 'line'})
    for bbox in text_boxes:
        all_regions.append({'bbox': bbox, 'polygon': None, 'kind': 'text'})

    mask, grey_mask, colour_mask, text_mask = build_colour_exact_mask(
        gray, img_bgr, all_regions, colour_lo_raw, colour_hi_raw
    )

    # Inpaint.
    cleaned = inpaint_masked(img_bgr, mask)

    debug_info = {
        'caliper_boxes_raw': caliper_boxes_raw,
        'caliper_boxes': caliper_boxes,
        'dotted_lines': dotted_lines,
        'traced_lines': traced_lines,
        'all_regions': all_regions,
        'colour_range': (colour_lo, colour_hi),
        'colour_range_raw': (colour_lo_raw, colour_hi_raw),
        'mask': mask,
        'grey_mask': grey_mask,
        'colour_mask': colour_mask,
        'text_mask': text_mask,
        'ocr_regions': len(text_boxes),
        'used_fallback': need_fallback
    }

    return cleaned, debug_info

# Visualisation.
def draw_debug_overlay(img_bgr, debug_info):
    """
    Colour coding (bgr):
    green (0,255,0) - raw caliper boxes
    blue (255,100,0) - expanded caliper boxes
    cyan (0,230,230) - paired dotted-line polygons
    magenta (255,0,255) - traced line polygons
    """
    out = img_bgr.copy()
    lo, hi = debug_info['colour_range']
    lo_r, hi_r = debug_info.get('colour_range_raw', (lo, hi))

    for x1, y1, x2, y2 in debug_info['caliper_boxes_raw']:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for x1, y1, x2, y2 in debug_info['caliper_boxes']:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), 1)
    for ln in debug_info['dotted_lines']:
        cv2.polylines(out, [ln['polygon'].astype(np.int32)], True, (0, 230, 230), 1)
    for ln in debug_info.get('traced_lines', []):
        cv2.polylines(out, [ln['polygon'].astype(np.int32)], True, (255, 0, 255), 1)

    cv2.putText(out, f"raw[{lo_r},{hi_r}] clamped[{lo},{hi}]",
                (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    n_cal = len(debug_info['caliper_boxes_raw'])
    n_lines = len(debug_info['dotted_lines'])
    n_traced = len(debug_info.get('traced_lines', []))
    cv2.putText(out, f"{n_cal} cal  {n_lines} lines  {n_traced} traced",
                (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out

if __name__ == '__main__':
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt

    paths = sys.argv[1:] if len(sys.argv) > 1 else ['malignant__15_.png']
    n = len(paths)

    fig, axes = plt.subplots(n, 4, figsize=(20, 5*n))
    if n == 1:
        axes = axes[None, :]

    for row, path in enumerate(paths):
        print(f'\nprocessing: {path}')
        img = cv2.imread(path)
        if img is None:
            print(f'  error: could not read {path!r}')
            continue

        cleaned, dbg = preprocess_image(img)

        lo, hi = dbg['colour_range']
        print(f'  colour range: [{lo},{hi}]')
        print(f'  calipers: {len(dbg["caliper_boxes_raw"])}')
        print(f'  dotted lines: {len(dbg["dotted_lines"])}')
        print(f'  traced lines: {len(dbg.get("traced_lines", []))}')
        for ln in dbg['dotted_lines']:
            print(f'    paired score={ln["score"]:.2f} bbox={ln["bbox"]}')
        for ln in dbg.get('traced_lines', []):
            print(f'    traced bbox={ln["bbox"]}')

        overlay = draw_debug_overlay(img, dbg)

        # Mask visualisation: white=grey annotation, red=colour annotation.
        mask_vis = np.zeros_like(img)
        mask_vis[dbg['grey_mask'] > 0] = (255, 255, 255)
        mask_vis[dbg['colour_mask'] > 0] = (0, 0, 255)

        for col, (title, frame) in enumerate([
            ('original', img),
            ('detections', overlay),
            ('mask', mask_vis),
            ('cleaned', cleaned),
        ]):
            axes[row, col].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'{Path(path).name}\n{title}', fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    out_path = 'preprocessing_results.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'\nsaved -> {out_path}')
    plt.show()