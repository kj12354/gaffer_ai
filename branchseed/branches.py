"""Ostium clustering, outward growth, eligibility filters, landmark estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass, distance_transform_edt, label
from skimage.segmentation import watershed

from .cross_section import path_consistency
from .intensity import IntensityRange
from .io import VolumePair, np_zyx_to_mm
from .params import PipelineParams
from .skeleton import ridge_walk


@dataclass
class Daughter:
    """One eligible direct aortic daughter."""

    ostium_zyx: np.ndarray
    seed_zyx: np.ndarray
    radius_mm: float
    direction_xyz: np.ndarray
    path_length_mm: float
    origin_diameter_mm: float
    n_voxels: int
    mean_hu: float = 0.0
    med_vesselness: float = 0.0
    extent_mm: float = 0.0
    radius_cv: float = float("nan")
    mean_circularity: float = float("nan")
    bone_frac: float = 0.0
    adj_lumen_frac: float = 0.0
    # Output-only landmarks for long wall-kissing contacts.
    output_ostium_zyx: np.ndarray | None = None
    output_seed_zyx: np.ndarray | None = None


@dataclass
class DetectionResult:
    daughters: list[Daughter] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    merge_log: list[str] = field(default_factory=list)


def _label_and_merge(
    seeds: np.ndarray,
    spacing_zyx: np.ndarray,
    merge_mm: float,
) -> np.ndarray:
    """Label 26-connected seed patches and merge those closer than ``merge_mm``."""
    labeled, n = label(seeds, structure=np.ones((3, 3, 3)))
    if n == 0:
        return labeled
    centroids = []
    for i in range(1, n + 1):
        com = center_of_mass(seeds, labeled, i)
        centroids.append(np.array(com, dtype=np.float64) * spacing_zyx)
    centroids = np.asarray(centroids)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(centroids[i] - centroids[j])) <= merge_mm:
                parent[find(j)] = find(i)
    remap = {}
    nxt = 1
    out = np.zeros_like(labeled)
    for i in range(1, n + 1):
        r = find(i - 1)
        if r not in remap:
            remap[r] = nxt
            nxt += 1
        out[labeled == i] = remap[r]
    return out


def _cap_slices(aorta: np.ndarray) -> tuple[set[int], bool, bool]:
    """z-indices of flat superior/inferior crop faces of the aorta."""
    zs = np.where(np.any(aorta, axis=(1, 2)))[0]
    if zs.size == 0:
        return set(), False, False
    zmin, zmax = int(zs[0]), int(zs[-1])
    # A "cap" is a face where the aorta occupies a large contiguous blob —
    # typical of a cropped tubular lumen — rather than a pointed terminus.
    def face_is_cap(z: int) -> bool:
        sl = aorta[z]
        return bool(sl.sum() >= 8)

    lo_cap = face_is_cap(zmin)
    hi_cap = face_is_cap(zmax)
    caps: set[int] = set()
    if lo_cap:
        caps.update(range(zmin, zmin + 2))
    if hi_cap:
        caps.update(range(max(zmax - 1, zmin), zmax + 1))
    return caps, lo_cap, hi_cap


def _drop_cap_seeds(seeds: np.ndarray, aorta: np.ndarray, margin: int) -> np.ndarray:
    caps, lo, hi = _cap_slices(aorta)
    if not caps:
        return seeds
    zs = np.where(np.any(aorta, axis=(1, 2)))[0]
    zmin, zmax = int(zs[0]), int(zs[-1])
    cleaned = seeds.copy()
    if lo:
        cleaned[: zmin + margin] = False
    if hi:
        cleaned[zmax - margin + 1 :] = False
    return cleaned


def grow_instances(
    seeds: np.ndarray,
    allowed: np.ndarray,
    dist: np.ndarray,
    spacing_zyx: np.ndarray,
    merge_mm: float,
) -> np.ndarray:
    """Assign allowed voxels to the nearest merged ostium seed (masked Voronoi)."""
    markers = _label_and_merge(seeds, spacing_zyx, merge_mm)
    if markers.max() == 0:
        return markers
    # EDT from the markers (0 on seeds, increasing away) yields a geodesic
    # Voronoi partition and does not flood around the wall the way -dist does.
    basins = distance_transform_edt(markers == 0, sampling=tuple(spacing_zyx))
    return watershed(basins, markers=markers, mask=allowed).astype(np.int32)


def _contact_spread_mm(contact: np.ndarray, spacing_zyx: np.ndarray) -> float:
    coords = np.argwhere(contact).astype(np.float64)
    if len(coords) == 0:
        return 0.0
    phys = coords * spacing_zyx.reshape(1, 3)
    return float(np.max(np.linalg.norm(phys - phys.mean(axis=0), axis=1)))


def _split_if_multiple_ostia(
    inst: np.ndarray,
    contact: np.ndarray,
    spacing_zyx: np.ndarray,
    merge_mm: float,
    min_sep_mm: float = 5.0,
    max_sep_mm: float = 12.0,
) -> list[np.ndarray]:
    """Split only a clear two-ostium contact, not a fragmented wall ring.

    Patches closer than ``merge_mm`` stay one origin. Noisy multi-fragment
    contacts are left intact so viscerals are not oversplit.
    """
    markers = _label_and_merge(contact, spacing_zyx, merge_mm)
    n = int(markers.max())
    if n != 2:
        return [inst]
    cents = []
    sizes = []
    for i in range(1, n + 1):
        pts = np.argwhere(markers == i).astype(np.float64)
        if len(pts) < 3:
            return [inst]
        cents.append((pts * spacing_zyx.reshape(1, 3)).mean(axis=0))
        sizes.append(len(pts))
    sep = float(np.linalg.norm(cents[0] - cents[1]))
    if sep < min_sep_mm or sep > max_sep_mm:
        return [inst]
    basins = distance_transform_edt(markers == 0, sampling=tuple(spacing_zyx))
    labels = watershed(basins, markers=markers, mask=inst)
    parts = [labels == i for i in range(1, int(labels.max()) + 1) if np.any(labels == i)]
    return parts if len(parts) == 2 else [inst]


def _contact_voxels(instance: np.ndarray, aorta: np.ndarray) -> np.ndarray:
    """Instance voxels that 26-touch the parent aorta."""
    dilated = binary_dilation(aorta, structure=np.ones((3, 3, 3)))
    return instance & dilated & ~aorta


def _ostium_zyx(contact: np.ndarray, aorta: np.ndarray) -> np.ndarray:
    """Ostium = centroid of parent-surface voxels that touch this instance."""
    if not np.any(contact):
        com = center_of_mass(aorta)
        return np.array(com, dtype=np.float64)
    # Map contact voxels onto the adjacent aortic surface.
    dilated_c = binary_dilation(contact, structure=np.ones((3, 3, 3)))
    surface = aorta & dilated_c
    target = surface if np.any(surface) else contact
    return np.array(center_of_mass(target), dtype=np.float64)


def _strip_landmarks(
    contact: np.ndarray,
    inst: np.ndarray,
    spacing_zyx: np.ndarray,
    min_path_mm: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ostium at the origin end of a long wall-kiss; seed ~5 mm along it.

    Alongside daughters kiss the aorta as a strip. The contact centroid then
    sits mid-vessel, so seed−ostium points back toward the true origin.
    Keep/reject still uses the centroid; this is output-only.
    """
    coords = np.argwhere(contact).astype(np.float64)
    if len(coords) < 6:
        return None
    sp = np.asarray(spacing_zyx, dtype=np.float64).reshape(1, 3)
    phys = coords * sp
    centered = phys - phys.mean(axis=0)
    try:
        _, ev, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if ev.size < 2:
        return None
    aniso = float(ev[0] / max(float(ev[1]), 1e-3))
    proj = centered @ vt[0]
    span = float(proj.max() - proj.min())
    if aniso < 1.80 or span < 5.0:
        return None
    i0, i1 = int(np.argmin(proj)), int(np.argmax(proj))
    end0, end1 = phys[i0], phys[i1]
    axis = end1 - end0
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-6:
        return None
    axis = axis / nrm
    inst_phys = np.argwhere(inst).astype(np.float64) * sp
    from0 = float(((inst_phys - end0) @ axis).max())
    from1 = float(((inst_phys - end1) @ (-axis)).max())
    if from0 >= from1:
        ost_phys, sign = end0, 1.0
    else:
        ost_phys, sign = end1, -1.0
    target = ost_phys + sign * float(min_path_mm) * axis
    nearest = inst_phys[int(np.argmin(((inst_phys - target) ** 2).sum(axis=1)))]
    scale = np.asarray(spacing_zyx, dtype=np.float64)
    return (ost_phys / scale).astype(np.float64), (nearest / scale).astype(np.float64)


def _origin_diameter_mm(
    contact: np.ndarray,
    lumen_radius: np.ndarray,
    spacing_zyx: np.ndarray,
) -> float:
    n = int(contact.sum())
    if n == 0:
        return 0.0
    voxel_area = float(np.prod(spacing_zyx) ** (2.0 / 3.0))
    area_diam = 2.0 * float(np.sqrt((n * voxel_area) / np.pi))
    local_r = float(lumen_radius[contact].max()) if np.any(contact) else 0.0
    return max(area_diam, 2.0 * local_r)


def _radius_at_seed(
    seed_zyx: np.ndarray,
    lumen: np.ndarray,
    direction_zyx_mm: np.ndarray,
    ct: np.ndarray,
    rng: IntensityRange,
    spacing_zyx: np.ndarray,
) -> float:
    """Local lumen radius at the seed via DT plus a perpendicular-plane fallback."""
    dt = distance_transform_edt(lumen, sampling=tuple(spacing_zyx))
    z, y, x = [int(round(c)) for c in seed_zyx]
    z = int(np.clip(z, 0, lumen.shape[0] - 1))
    y = int(np.clip(y, 0, lumen.shape[1] - 1))
    x = int(np.clip(x, 0, lumen.shape[2] - 1))
    r_dt = float(dt[z, y, x])
    r_plane = _plane_radius(seed_zyx, direction_zyx_mm, ct, rng, spacing_zyx)
    # Distance transform of a 1-voxel tube underestimates slightly; take the
    # more conservative positive estimate rather than inflating.
    estimates = [r for r in (r_dt, r_plane) if r > 0]
    if not estimates:
        return max(r_dt, 0.5 * float(np.mean(spacing_zyx)))
    return float(np.median(estimates))


def _plane_radius(
    seed_zyx: np.ndarray,
    direction_zyx_mm: np.ndarray,
    ct: np.ndarray,
    rng: IntensityRange,
    spacing_zyx: np.ndarray,
    half_extent_mm: float = 4.0,
) -> float:
    """Area-equivalent radius on a coarse plane perpendicular to ``direction``."""
    d = np.asarray(direction_zyx_mm, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-8:
        return 0.0
    d = d / nrm
    # Build two in-plane axes.
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a1 = np.cross(d, helper)
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(d, a1)
    step = 0.5 * float(np.min(spacing_zyx))
    n = int(np.ceil(half_extent_mm / step))
    count = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if (i * step) ** 2 + (j * step) ** 2 > half_extent_mm**2:
                continue
            pos = seed_zyx + (i * step * a1 + j * step * a2) / spacing_zyx
            zi, yi, xi = [int(round(c)) for c in pos]
            if (
                0 <= zi < ct.shape[0]
                and 0 <= yi < ct.shape[1]
                and 0 <= xi < ct.shape[2]
                and rng.blood_low <= ct[zi, yi, xi] <= rng.blood_high
            ):
                count += 1
    area = count * (step**2)
    if area <= 0:
        return 0.0
    return float(np.sqrt(area / np.pi))


def _direction_xyz(direction_zyx_mm: np.ndarray) -> np.ndarray:
    """Map a physical (z,y,x) unit vector onto SimpleITK (x,y,z) LPS."""
    dz, dy, dx = [float(c) for c in direction_zyx_mm]
    vec = np.array([dx, dy, dz], dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / n


def _is_crop_cap_origin(
    ostium_zyx: np.ndarray,
    direction_xyz: np.ndarray,
    aorta: np.ndarray,
    params: PipelineParams,
) -> bool:
    caps, lo, hi = _cap_slices(aorta)
    z = int(round(float(ostium_zyx[0])))
    if z not in caps and not any(abs(z - c) <= params.cap_slice_margin for c in caps):
        return False
    # Reject only when the path dives through the flat cropped end.
    return abs(float(direction_xyz[2])) >= params.cap_dir_z_abs


def _is_iliac(
    ostium_zyx: np.ndarray,
    direction_xyz: np.ndarray,
    aorta: np.ndarray,
    spacing_zyx: np.ndarray,
    image,
    origin_zyx: np.ndarray,
    params: PipelineParams,
) -> bool:
    """Reject the terminal iliac split, not side branches such as the IMA.

    LPS +Z is superior. The inferior terminus is the set of aorta voxels
    with the smallest physical Z. A large inferior-going vessel that leaves
    within a few millimetres of that terminus is treated as an iliac.
    """
    coords = np.argwhere(aorta)
    if coords.size == 0:
        return False
    # Physical Z of every aorta voxel (crop coordinates → full → LPS).
    full = coords.astype(np.float64) + origin_zyx.astype(np.float64)
    phys_z = np.array(
        [float(np_zyx_to_mm(image, z, y, x)[2]) for z, y, x in full[:: max(1, len(full) // 400)]]
    )
    if phys_z.size == 0:
        return False
    z_inf = float(phys_z.min())
    ostium_full = ostium_zyx + origin_zyx.astype(np.float64)
    ostium_z = float(np_zyx_to_mm(image, *ostium_full)[2])
    if ostium_z > z_inf + params.iliac_inferior_window_mm:
        return False
    if float(direction_xyz[2]) > params.iliac_dir_z:
        return False
    return (ostium_z - z_inf) <= params.iliac_terminus_mm


def _instance_quality(
    inst: np.ndarray,
    contact: np.ndarray,
    ct: np.ndarray,
    vesselness: np.ndarray,
    v_thresh: float,
    rng: IntensityRange,
    spacing_zyx: np.ndarray,
    params: PipelineParams,
) -> str | None:
    """Return a reject reason, or None if the instance looks like a tube."""
    coords = np.argwhere(inst).astype(np.float64)
    if len(coords) < 4:
        return "too_few_voxels"
    phys = coords * spacing_zyx.reshape(1, 3)
    centered = phys - phys.mean(axis=0)
    try:
        evals = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        evals = np.array([0.0, 0.0, 0.0])
    if evals[0] < 3.2:
        return f"short_axis_{evals[0]:.1f}"
    aniso = float(evals[0] / max(evals[1], 1e-3))
    if aniso < 1.15:
        return f"blob_aniso_{aniso:.2f}"

    mean_hu = float(ct[inst].mean())
    hu_floor = min(rng.p10, 0.45 * rng.p50 + 0.55 * rng.blood_low)
    if mean_hu < hu_floor:
        return f"dim_hu_{mean_hu:.0f}"

    med_v = float(np.median(vesselness[inst]))
    bright_enough = mean_hu >= 0.7 * rng.p50
    if (not bright_enough) and med_v < 0.4 * v_thresh:
        return f"low_vesselness_{med_v:.3f}"

    n = int(inst.sum())
    area_diam = _origin_diameter_mm(contact, np.zeros(inst.shape, dtype=np.float32), spacing_zyx)
    # Giant low-vesselness blobs are search-shell smears, not tubes.
    if med_v < 0.02 and n > 400 and area_diam < 6.5:
        return f"smear_v{med_v:.3f}_n{n}"
    voxel_vol = float(np.prod(spacing_zyx))
    if n * voxel_vol < 20.0:
        return "tiny_volume"
    return None


def _ostium_span_mm(
    inst: np.ndarray,
    ostium_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
) -> float:
    """Furthest instance voxel from the ostium, in millimetres."""
    coords = np.argwhere(inst)
    if coords.size == 0:
        return 0.0
    delta = (coords.astype(np.float64) - ostium_zyx.reshape(1, 3)) * spacing_zyx.reshape(1, 3)
    return float(np.sqrt((delta * delta).sum(axis=1)).max())


def _ostium_context(
    ostium_zyx: np.ndarray,
    direction_zyx: np.ndarray,
    ct: np.ndarray,
    aorta: np.ndarray,
    blood: np.ndarray,
    rng: IntensityRange,
    spacing_zyx: np.ndarray,
    params: PipelineParams,
) -> tuple[float, float]:
    """Bone and leftover-lumen fractions in an outward ball at the ostium.

    Real daughters occupy a thin outward tube. Vertebra contact lights up
    high-HU voxels; IVC / blob contact leaves blood outside that tube.
    """
    radius_mm = float(params.context_radius_mm)
    tube_r = float(params.tube_clear_radius_mm)
    d = np.asarray(direction_zyx, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-8:
        return 0.0, 0.0
    d = d / nrm

    rz = int(np.ceil(radius_mm / float(spacing_zyx[0]))) + 1
    ry = int(np.ceil(radius_mm / float(spacing_zyx[1]))) + 1
    rx = int(np.ceil(radius_mm / float(spacing_zyx[2]))) + 1
    z0, y0, x0 = [float(c) for c in ostium_zyx]
    z1, z2 = max(0, int(z0) - rz), min(ct.shape[0], int(z0) + rz + 1)
    y1, y2 = max(0, int(y0) - ry), min(ct.shape[1], int(y0) + ry + 1)
    x1, x2 = max(0, int(x0) - rx), min(ct.shape[2], int(x0) + rx + 1)
    if z2 <= z1 or y2 <= y1 or x2 <= x1:
        return 0.0, 0.0

    zz, yy, xx = np.mgrid[z1:z2, y1:y2, x1:x2]
    dz = (zz.astype(np.float64) - z0) * float(spacing_zyx[0])
    dy = (yy.astype(np.float64) - y0) * float(spacing_zyx[1])
    dx = (xx.astype(np.float64) - x0) * float(spacing_zyx[2])
    dist = np.sqrt(dz * dz + dy * dy + dx * dx)
    axial = dz * d[0] + dy * d[1] + dx * d[2]
    radial2 = dist * dist - axial * axial
    ball = dist <= radius_mm
    outside = ball & ~aorta[z1:z2, y1:y2, x1:x2]
    outward = axial >= -1.0
    region = outside & outward
    if not np.any(region):
        return 0.0, 0.0

    hu = ct[z1:z2, y1:y2, x1:x2][region]
    bone_hu = max(float(rng.blood_high), float(rng.p95) + 40.0)
    bone_frac = float(np.mean(hu >= bone_hu))

    leftover = (
        blood[z1:z2, y1:y2, x1:x2]
        & ~aorta[z1:z2, y1:y2, x1:x2]
        & ball
        & outward
        & (radial2 > tube_r * tube_r)
    )
    voxel_vol = float(np.prod(spacing_zyx))
    leftover_mm3 = float(leftover.sum()) * voxel_vol
    ball_mm3 = (2.0 / 3.0) * np.pi * radius_mm**3
    adj = leftover_mm3 / max(ball_mm3, 1.0)
    return bone_frac, float(min(1.0, adj))


def _seed_at_distance_zyx(
    ostium_zyx: np.ndarray,
    seed_zyx: np.ndarray,
    direction_zyx: np.ndarray | None,
    spacing_zyx: np.ndarray,
    target_mm: float,
) -> np.ndarray:
    """Place the seed ``target_mm`` from the ostium along the current heading.

    Challenge definition: seed is 5 mm outward in the daughter lumen.
    """
    ost = np.asarray(ostium_zyx, dtype=np.float64)
    seed = np.asarray(seed_zyx, dtype=np.float64)
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    vec = (seed - ost) * sp
    n = float(np.linalg.norm(vec))
    if n > 1e-6:
        direction = vec / n
    elif direction_zyx is not None:
        d = np.asarray(direction_zyx, dtype=np.float64)
        dn = float(np.linalg.norm(d))
        direction = d / dn if dn > 1e-6 else np.array([1.0, 0.0, 0.0])
    else:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    target = ost + direction * float(target_mm) / sp
    return target


def _likely_artifact(d: Daughter, params: PipelineParams) -> str | None:
    """Conservative reject reasons from crop-adjudicated false patterns."""
    if float(d.bone_frac) >= params.bone_frac_drop:
        return f"bone_frac_{d.bone_frac:.2f}"
    circ = float(d.mean_circularity)
    if (
        float(d.path_length_mm) < params.short_path_drop_mm
        and np.isfinite(circ)
        and circ < params.short_circ_max
    ):
        return f"short_flat_{d.path_length_mm:.1f}"
    if (
        float(d.med_vesselness) < params.blob_vess_max
        and float(d.adj_lumen_frac) >= params.adj_lumen_drop
        and float(d.radius_mm) >= params.blob_radius_mm
    ):
        return f"adj_blob_v{d.med_vesselness:.3f}"
    cv = float(d.radius_cv)
    if (
        float(d.med_vesselness) < params.smear_vess_max
        and np.isfinite(cv)
        and cv >= params.smear_radius_cv
    ):
        return f"smear_cv_{cv:.2f}"
    return None


def extract_daughters(
    vol: VolumePair,
    ct: np.ndarray,
    aorta: np.ndarray,
    search: dict[str, np.ndarray],
    rng: IntensityRange,
    origin_zyx: np.ndarray,
    params: PipelineParams,
) -> DetectionResult:
    """Turn vesselness / blood maps into a de-duplicated daughter list."""
    result = DetectionResult()
    seeds = _drop_cap_seeds(search["seeds"], aorta, params.cap_slice_margin)
    if not np.any(seeds):
        result.rejected.append("no_seeds")
        return result

    instances = grow_instances(
        seeds, search["allowed"], search["dist"], vol.spacing_zyx, params.ostium_merge_mm
    )
    lumen_radius = distance_transform_edt(
        search["blood"] | aorta, sampling=tuple(vol.spacing_zyx)
    ).astype(np.float32)

    pieces: list[np.ndarray] = []
    for inst_id in range(1, int(instances.max()) + 1):
        inst = instances == inst_id
        contact = _contact_voxels(inst, aorta)
        if not np.any(contact):
            result.rejected.append(f"id{inst_id}:no_parent_contact")
            continue
        pieces.append(inst)

    candidates: list[Daughter] = []
    for inst_id, inst in enumerate(pieces, start=1):
        contact = _contact_voxels(inst, aorta)
        if not np.any(contact):
            result.rejected.append(f"id{inst_id}:no_parent_contact")
            continue

        ostium = _ostium_zyx(contact, aorta)
        diam = _origin_diameter_mm(contact, lumen_radius, vol.spacing_zyx)
        extent_mm = float(search["dist"][inst].max()) if np.any(inst) else 0.0
        span_mm = _ostium_span_mm(inst, ostium, vol.spacing_zyx)
        path = ridge_walk(
            ostium,
            inst,
            search["dist"],
            lumen_radius,
            vol.spacing_zyx,
            max_mm=params.max_trace_mm,
            min_mm=params.min_path_mm,
        )
        # Eligibility is ≥5 mm of followable lumen beyond the wall.
        # Radial wall-distance under-counts daughters that run alongside
        # the aorta; span from the ostium covers that case. Stored path
        # length stays ridge/extent so quality rules are not inflated.
        followable_mm = max(float(path.length_mm), float(extent_mm), float(span_mm))
        reported_mm = max(float(path.length_mm), float(extent_mm))
        if followable_mm < params.min_path_mm:
            result.rejected.append(
                f"id{inst_id}:short_path_{path.length_mm:.1f}_ext_{extent_mm:.1f}_span_{span_mm:.1f}mm"
            )
            continue
        if reported_mm < params.min_path_mm:
            # Span-only pass: must be a long thin tube, not a fat wall blob.
            coords = np.argwhere(inst).astype(np.float64) * vol.spacing_zyx.reshape(1, 3)
            centered = coords - coords.mean(axis=0)
            try:
                ev = np.linalg.svd(centered, compute_uv=False)
            except np.linalg.LinAlgError:
                ev = np.zeros(3)
            aniso = float(ev[0] / max(float(ev[1]), 1e-3))
            med_v = float(np.median(search["vesselness"][inst]))
            if aniso < 1.50 or float(ev[0]) < 5.0 or med_v < 0.05:
                result.rejected.append(
                    f"id{inst_id}:span_blob_aniso_{aniso:.2f}_v{med_v:.3f}"
                )
                continue
        if (not path.seed_ok) or path.length_mm < params.min_path_mm:
            # Place the seed at ≥5 mm along the instance when the ridge ended early.
            far = inst & (search["dist"] >= params.min_path_mm)
            if not np.any(far):
                coords = np.argwhere(inst)
                ost = ostium.reshape(1, 3)
                d2 = ((coords - ost) * vol.spacing_zyx.reshape(1, 3)) ** 2
                far_idx = d2.sum(axis=1) >= (params.min_path_mm ** 2)
                far = np.zeros(inst.shape, dtype=bool)
                if np.any(far_idx):
                    pick = coords[far_idx]
                    far[tuple(pick.T)] = True
            if np.any(far):
                coords = np.argwhere(far)
                ost = ostium.reshape(1, 3)
                delta = (coords - ost) * vol.spacing_zyx.reshape(1, 3)
                d2 = (delta * delta).sum(axis=1)
                dist_at = search["dist"][tuple(coords.T)]
                # Prefer ~5 mm from the ostium, then more outward from the wall.
                score = np.abs(d2 - params.min_path_mm ** 2) - 2.0 * dist_at
                path.seed_zyx = coords[int(np.argmin(score))].astype(np.float64)
                path.seed_ok = True
                step = (path.seed_zyx - ostium) * vol.spacing_zyx
                n = float(np.linalg.norm(step))
                if n > 1e-6:
                    path.direction_zyx = step / n
                path.length_mm = reported_mm

        direction_xyz = _direction_xyz(path.direction_zyx)
        if _is_crop_cap_origin(ostium, direction_xyz, aorta, params):
            result.rejected.append(f"id{inst_id}:crop_cap")
            continue
        if (
            diam >= 4.5
            and _is_iliac(ostium, direction_xyz, aorta, vol.spacing_zyx, vol.image, origin_zyx, params)
        ):
            result.rejected.append(f"id{inst_id}:iliac")
            continue

        radius = _radius_at_seed(
            path.seed_zyx, inst, path.direction_zyx, ct, rng, vol.spacing_zyx
        )
        if diam < params.min_origin_diameter_mm and radius < params.min_seed_radius_mm:
            result.rejected.append(
                f"id{inst_id}:tiny_diam_{diam:.2f}_r_{radius:.2f}"
            )
            continue

        quality = _instance_quality(
            inst, contact, ct, search["vesselness"], float(search["v_thresh"]),
            rng, vol.spacing_zyx, params,
        )
        if quality is not None:
            result.rejected.append(f"id{inst_id}:{quality}")
            continue

        path_pts = np.vstack([ostium.reshape(1, 3), path.points_zyx, path.seed_zyx.reshape(1, 3)])
        try:
            radius_cv, mean_circ = path_consistency(path_pts, inst, vol.spacing_zyx)
        except Exception:
            radius_cv, mean_circ = float("nan"), float("nan")

        bone_frac, adj_frac = _ostium_context(
            ostium,
            path.direction_zyx,
            ct,
            aorta,
            search["blood"],
            rng,
            vol.spacing_zyx,
            params,
        )

        # Span-only alongside tubes: snap output ostium to the origin end
        # of the wall-kiss. Radially leaving viscerals keep the centroid.
        strip = None
        if reported_mm < params.min_path_mm:
            strip = _strip_landmarks(
                contact, inst, vol.spacing_zyx, params.min_path_mm
            )
        out_ost, out_seed = (strip if strip is not None else (None, None))

        candidates.append(
            Daughter(
                ostium_zyx=ostium,
                seed_zyx=path.seed_zyx,
                radius_mm=float(max(radius, 0.5 * float(np.min(vol.spacing_zyx)))),
                direction_xyz=direction_xyz,
                path_length_mm=float(reported_mm),
                origin_diameter_mm=float(diam),
                n_voxels=int(inst.sum()),
                mean_hu=float(ct[inst].mean()),
                med_vesselness=float(np.median(search["vesselness"][inst])),
                extent_mm=float(extent_mm),
                radius_cv=float(radius_cv),
                mean_circularity=float(mean_circ),
                bone_frac=float(bone_frac),
                adj_lumen_frac=float(adj_frac),
                output_ostium_zyx=out_ost,
                output_seed_zyx=out_seed,
            )
        )

    # 1) Unconditional near-duplicate collapse (same ostium, two seeds).
    # 2) Path-divergence merge for genuinely distinct-but-nearby ostia.
    kept, near_log = _merge_near_duplicates(
        candidates, vol, origin_zyx, params, vol.case_id
    )
    kept, path_log = _dedup_ostia(kept, vol, origin_zyx, params)
    result.merge_log = near_log + path_log
    if params.quality_filter:
        filtered: list[Daughter] = []
        for d in kept:
            reason = _likely_artifact(d, params)
            if reason is not None:
                result.rejected.append(f"quality:{reason}")
                continue
            filtered.append(d)
        kept = filtered
    for d in kept:
        if d.output_ostium_zyx is not None:
            d.ostium_zyx = d.output_ostium_zyx
        seed_src = d.output_seed_zyx if d.output_seed_zyx is not None else d.seed_zyx
        d.seed_zyx = _seed_at_distance_zyx(
            d.ostium_zyx,
            seed_src,
            None,
            vol.spacing_zyx,
            params.min_path_mm,
        )
        step = (d.seed_zyx - d.ostium_zyx) * vol.spacing_zyx
        n = float(np.linalg.norm(step))
        if n > 1e-6:
            d.direction_xyz = _direction_xyz(step)
    # Stable order: superior → inferior (LPS +Z), then +X.
    kept.sort(
        key=lambda d: (
            -float(np_zyx_to_mm(vol.image, *(d.ostium_zyx + origin_zyx))[2]),
            float(np_zyx_to_mm(vol.image, *(d.ostium_zyx + origin_zyx))[0]),
        )
    )
    result.daughters = kept
    return result


def _near_dup_score(d: Daughter) -> float:
    """Prefer the longer followable path, then higher vesselness."""
    return float(d.path_length_mm) + 20.0 * float(d.med_vesselness)


def _merge_near_duplicates(
    candidates: list[Daughter],
    vol: VolumePair,
    origin_zyx: np.ndarray,
    params: PipelineParams,
    case_id: str,
) -> tuple[list[Daughter], list[str]]:
    """Collapse ostia closer than ``near_dup_ostium_mm`` regardless of direction.

    This is a different rule from path-divergence merge: at a few millimetres
    two detections are the same physical origin, so heading is not evidence
    of two branches.
    """
    log: list[str] = []
    if len(candidates) <= 1:
        return candidates, log
    ostia = np.array(
        [np_zyx_to_mm(vol.image, *(c.ostium_zyx + origin_zyx)) for c in candidates]
    )
    keep = [True] * len(candidates)
    floor = float(params.near_dup_ostium_mm)
    for i in range(len(candidates)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(candidates)):
            if not keep[j]:
                continue
            d_ost = float(np.linalg.norm(ostia[i] - ostia[j]))
            if d_ost > floor:
                continue
            a, b = candidates[i], candidates[j]
            if _near_dup_score(a) >= _near_dup_score(b):
                keep[j] = False
                kept_i, dropped = i, j
            else:
                keep[i] = False
                kept_i, dropped = j, i
            log.append(
                f"near_dup MERGE {case_id} 3D={d_ost:.2f}mm "
                f"keep_idx={kept_i} drop_idx={dropped} "
                f"keep_z={ostia[kept_i][2]:.1f} drop_z={ostia[dropped][2]:.1f} "
                f"vess=({a.med_vesselness:.3f},{b.med_vesselness:.3f}) "
                f"path=({a.path_length_mm:.1f},{b.path_length_mm:.1f})"
            )
            if not keep[i]:
                break
    return [c for c, k in zip(candidates, keep) if k], log


def _dedup_ostia(
    candidates: list[Daughter],
    vol: VolumePair,
    origin_zyx: np.ndarray,
    params: PipelineParams,
) -> tuple[list[Daughter], list[str]]:
    """Merge duplicate seeds of one origin; keep ostia whose paths diverge.

    Close ostia (≤ ``ostium_merge_mm``) always collapse. Ostia out to
    ``path_merge_ostium_mm`` collapse only when the traced directions agree
    or the seeds converge. Diverging headings are left as two instances.
    """
    log: list[str] = []
    if len(candidates) <= 1:
        return candidates, log
    ostia = np.array(
        [np_zyx_to_mm(vol.image, *(c.ostium_zyx + origin_zyx)) for c in candidates]
    )
    seeds = np.array(
        [np_zyx_to_mm(vol.image, *(c.seed_zyx + origin_zyx)) for c in candidates]
    )
    keep = [True] * len(candidates)
    merge_mm = params.ostium_merge_mm
    for i in range(len(candidates)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(candidates)):
            if not keep[j]:
                continue
            d_ost = float(np.linalg.norm(ostia[i] - ostia[j]))
            d_seed = float(np.linalg.norm(seeds[i] - seeds[j]))
            cos = float(
                np.clip(
                    np.dot(candidates[i].direction_xyz, candidates[j].direction_xyz),
                    -1.0,
                    1.0,
                )
            )
            dz = abs(float(ostia[i][2] - ostia[j][2]))
            if d_ost <= merge_mm:
                reason = f"close_ostium_{d_ost:.1f}mm"
            elif d_ost > params.path_merge_ostium_mm:
                log.append(
                    f"pair {i},{j}: keep (3D ostium {d_ost:.1f} mm > "
                    f"{params.path_merge_ostium_mm:.1f} mm window; "
                    f"Δz={dz:.1f} mm dir_cos={cos:.2f} seed_sep={d_seed:.1f} mm)"
                )
                continue
            elif cos < params.path_diverge_dir_cosine or d_seed > d_ost + 2.0:
                log.append(
                    f"pair {i},{j}: keep_diverged (3D ostium {d_ost:.1f} mm "
                    f"dir_cos={cos:.2f} seed_sep={d_seed:.1f} mm Δz={dz:.1f} mm)"
                )
                continue
            elif cos < params.path_merge_min_dir_cosine and d_seed + 1.0 >= d_ost:
                log.append(
                    f"pair {i},{j}: keep_unaligned (3D ostium {d_ost:.1f} mm "
                    f"dir_cos={cos:.2f} seed_sep={d_seed:.1f} mm)"
                )
                continue
            else:
                reason = (
                    f"path_merge ostium={d_ost:.1f}mm dir_cos={cos:.2f} "
                    f"seed_sep={d_seed:.1f}mm"
                )
            a, b = candidates[i], candidates[j]
            score_a = a.path_length_mm + 0.3 * a.origin_diameter_mm
            score_b = b.path_length_mm + 0.3 * b.origin_diameter_mm
            if score_a >= score_b:
                keep[j] = False
                dropped, kept_i = j, i
            else:
                keep[i] = False
                dropped, kept_i = i, j
            log.append(
                f"pair {i},{j}: MERGE [{reason}] keep={kept_i} drop={dropped}"
            )
            if not keep[i]:
                break
    return [c for c, k in zip(candidates, keep) if k], log
