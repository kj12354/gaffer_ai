"""SimpleITK I/O and LPS physical-coordinate conversion.

All output landmarks go through SimpleITK
``TransformContinuousIndexToPhysicalPoint`` /
``TransformIndexToPhysicalPoint``. NiBabel affines are never used — they
typically return RAS and would silently invert the first two axes.
"""

from __future__ import annotations

import gzip
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk

# ITK rejects NIfTI sforms with a small shear (gantry tilt / resample).
# Permissive mode orthogonalizes on read when the installed ITK supports it.
os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "YES")

_ORTHO_ATOL = 1e-4
_NIFTI_DTYPE = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


@dataclass
class VolumePair:
    """Co-registered CT and binary aorta mask."""

    image: sitk.Image
    mask: sitk.Image
    ct: np.ndarray  # float32, numpy order (z, y, x)
    aorta: np.ndarray  # bool, (z, y, x)
    spacing_zyx: np.ndarray  # mm, aligned with numpy axes
    spacing_xyz: np.ndarray  # mm, SimpleITK index order
    case_id: str
    image_path: Path
    mask_path: Path

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.ct.shape)


def infer_case_id(image_path: str | Path, explicit: str | None = None) -> str:
    """Derive a stable case id such as ``subject019`` from the path."""
    if explicit:
        return explicit
    path = Path(image_path)
    folder = path.parent.name
    if folder.startswith("subject"):
        return folder
    if folder.startswith("case_"):
        try:
            return f"subject{int(folder.split('_')[1]):03d}"
        except (IndexError, ValueError):
            pass
    match = re.search(r"(\d+)", path.name)
    if match:
        return f"subject{int(match.group(1)):03d}"
    return path.stem.replace(".nii", "")


def _geometry_tuple(img: sitk.Image) -> tuple:
    return (
        tuple(img.GetSize()),
        tuple(round(s, 6) for s in img.GetSpacing()),
        tuple(round(o, 4) for o in img.GetOrigin()),
        tuple(round(d, 6) for d in img.GetDirection()),
    )


def _direction_matrix(img: sitk.Image) -> np.ndarray:
    return np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)


def _is_orthonormal(mat: np.ndarray, atol: float = _ORTHO_ATOL) -> bool:
    gram = mat.T @ mat
    return bool(np.allclose(gram, np.eye(3), atol=atol)) and abs(float(np.linalg.det(mat))) > 0.5


def _nearest_orthonormal(mat: np.ndarray) -> tuple[np.ndarray, float]:
    """SVD projection onto SO(3). Returns (R, frobenius ||M-R||)."""
    u, _s, vt = np.linalg.svd(mat)
    r = u @ vt
    if float(np.linalg.det(r)) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    mag = float(np.linalg.norm(mat - r, ord="fro"))
    return r, mag


def _apply_direction(img: sitk.Image, direction: np.ndarray) -> sitk.Image:
    out = sitk.Image(img)
    out.SetDirection(tuple(float(x) for x in np.asarray(direction).reshape(9)))
    return out


def _read_nifti_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.name.endswith(".gz") or raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def _parse_nifti1_affine(hdr: bytes) -> tuple[np.ndarray, tuple[int, int, int], np.dtype, float, float, float]:
    """Return (RAS 4x4 affine, size_xyz, dtype, vox_offset, slope, inter)."""
    endian = "<" if struct.unpack_from("<i", hdr, 0)[0] == 348 else ">"
    dim = struct.unpack_from(endian + "8h", hdr, 40)
    nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3] if dim[0] >= 3 else 1)
    datatype = struct.unpack_from(endian + "h", hdr, 70)[0]
    pixdim = struct.unpack_from(endian + "8f", hdr, 76)
    vox_offset = float(struct.unpack_from(endian + "f", hdr, 108)[0])
    slope = float(struct.unpack_from(endian + "f", hdr, 112)[0])
    inter = float(struct.unpack_from(endian + "f", hdr, 116)[0])
    qform_code = int(struct.unpack_from(endian + "h", hdr, 252)[0])
    sform_code = int(struct.unpack_from(endian + "h", hdr, 254)[0])
    if sform_code > 0:
        sx = struct.unpack_from(endian + "4f", hdr, 280)
        sy = struct.unpack_from(endian + "4f", hdr, 296)
        sz = struct.unpack_from(endian + "4f", hdr, 312)
        affine = np.array([sx, sy, sz, [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    elif qform_code > 0:
        b, c, d = struct.unpack_from(endian + "3f", hdr, 256)
        qx, qy, qz = struct.unpack_from(endian + "3f", hdr, 268)
        a = float(np.sqrt(max(1.0 - (b * b + c * c + d * d), 0.0)))
        r = np.array(
            [
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
            ],
            dtype=np.float64,
        )
        qfac = float(pixdim[0]) if pixdim[0] != 0 else 1.0
        affine = np.eye(4, dtype=np.float64)
        affine[:3, 0] = r[:, 0] * float(pixdim[1])
        affine[:3, 1] = r[:, 1] * float(pixdim[2])
        affine[:3, 2] = r[:, 2] * float(pixdim[3]) * qfac
        affine[:3, 3] = (qx, qy, qz)
    else:
        affine = np.diag([float(pixdim[1]), float(pixdim[2]), float(pixdim[3]), 1.0])
    dtype = _NIFTI_DTYPE.get(int(datatype))
    if dtype is None:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}")
    return affine, (nx, ny, nz), dtype, max(vox_offset, 352.0), slope, inter


def _ras_affine_to_lps_sitk(affine_ras: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """NIfTI RAS affine → SimpleITK (spacing, origin, direction) in LPS."""
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine = flip @ affine_ras
    a = affine[:3, :3]
    origin = tuple(float(x) for x in affine[:3, 3])
    spacing = tuple(float(np.linalg.norm(a[:, i])) for i in range(3))
    direction = np.eye(3, dtype=np.float64)
    for i, s in enumerate(spacing):
        if s > 1e-8:
            direction[:, i] = a[:, i] / s
    return spacing, origin, tuple(float(x) for x in direction.reshape(9))


def _read_nifti_bypass(path: Path) -> sitk.Image:
    """Load a NIfTI volume when SimpleITK/ITK rejects a sheared sform."""
    blob = _read_nifti_bytes(path)
    if len(blob) < 348:
        raise ValueError(f"File too small to be NIfTI: {path}")
    affine, size_xyz, dtype, offset, slope, inter = _parse_nifti1_affine(blob[:348])
    nx, ny, nz = size_xyz
    n = nx * ny * nz
    data = np.frombuffer(blob, dtype=dtype, count=n, offset=int(offset)).copy()
    data = data.reshape((nz, ny, nx))
    if slope != 0.0:
        data = data.astype(np.float32, copy=False) * np.float32(slope) + np.float32(inter)
    img = sitk.GetImageFromArray(data)
    spacing, origin, direction = _ras_affine_to_lps_sitk(affine)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img


def read_image_robust(path: str | Path, label: str = "image") -> tuple[sitk.Image, str]:
    """Read a volume, falling back if ITK rejects a non-orthonormal NIfTI sform."""
    path = Path(path)
    try:
        return sitk.ReadImage(str(path)), "sitk"
    except RuntimeError as exc:
        msg = str(exc)
        if "orthonormal" not in msg.lower() and "sform" not in msg.lower():
            raise
        os.environ["ITK_NIFTI_SFORM_PERMISSIVE"] = "YES"
        try:
            return sitk.ReadImage(str(path)), "sitk_permissive"
        except RuntimeError:
            img = _read_nifti_bypass(path)
            return img, f"nifti_bypass:{label}"


def _orthonormalize_pair(
    image: sitk.Image,
    mask: sitk.Image,
    case_id: str,
    log=print,
) -> tuple[sitk.Image, sitk.Image]:
    """Project both volumes onto one orthonormal physical grid."""
    d_img = _direction_matrix(image)
    d_msk = _direction_matrix(mask)
    need = (not _is_orthonormal(d_img)) or (not _is_orthonormal(d_msk))
    # Use the CT as the reference grid; apply the same rotation to the mask.
    r_img, mag_img = _nearest_orthonormal(d_img)
    r_msk, mag_msk = _nearest_orthonormal(d_msk)
    mag = max(mag_img, mag_msk)
    off_img = float(np.max(np.abs(d_img.T @ d_img - np.eye(3))))
    off_msk = float(np.max(np.abs(d_msk.T @ d_msk - np.eye(3))))
    if need or mag > _ORTHO_ATOL:
        image = _apply_direction(image, r_img)
        # Identical physical frame: CT's corrected direction + CT origin/spacing.
        mask = sitk.Image(mask)
        mask.SetDirection(image.GetDirection())
        mask.SetOrigin(image.GetOrigin())
        mask.SetSpacing(image.GetSpacing())
        if mask.GetSize() != image.GetSize():
            mask = sitk.Resample(
                mask,
                image,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                mask.GetPixelID(),
            )
        log(
            f"[{case_id}] non-orthonormal direction corrected  "
            f"frobenius={mag:.6f}  max_abs(DtD-I)={max(off_img, off_msk):.6f}  "
            f"det_before=({np.linalg.det(d_img):.6f}, {np.linalg.det(d_msk):.6f})"
        )
    if _geometry_tuple(image) != _geometry_tuple(mask):
        mask = sitk.Resample(
            mask,
            image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            mask.GetPixelID(),
        )
        log(f"[{case_id}] resampled mask onto corrected CT grid")
    if _geometry_tuple(image) != _geometry_tuple(mask):
        raise ValueError(
            "CT and aorta mask are not co-registered after direction correction: "
            f"image {_geometry_tuple(image)} vs mask {_geometry_tuple(mask)}"
        )
    if not _is_orthonormal(_direction_matrix(image)):
        raise ValueError(f"{case_id}: direction still not orthonormal after correction")
    return image, mask


def _log_header_shear(path: Path, case_id: str) -> None:
    """Report how far the on-disk NIfTI sform is from orthonormal (LPS)."""
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return
    try:
        blob = _read_nifti_bytes(path)
        affine, *_rest = _parse_nifti1_affine(blob[:348])
        _spacing, _origin, direction = _ras_affine_to_lps_sitk(affine)
        mat = np.asarray(direction, dtype=np.float64).reshape(3, 3)
        if _is_orthonormal(mat):
            return
        _r, mag = _nearest_orthonormal(mat)
        off = float(np.max(np.abs(mat.T @ mat - np.eye(3))))
        print(
            f"[{case_id}] on-disk NIfTI direction is non-orthonormal  "
            f"frobenius_to_SO3={mag:.6f}  max_abs(DtD-I)={off:.6f}  "
            f"det={float(np.linalg.det(mat)):.6f}"
        )
    except Exception:
        return


def load_case(
    image_path: str | Path,
    mask_path: str | Path,
    case_id: str | None = None,
) -> VolumePair:
    """Load a CT / aorta-mask pair and confirm they share one grid."""
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Aorta mask not found: {mask_path}")
    cid = infer_case_id(image_path, case_id)
    try:
        image, src_i = read_image_robust(image_path, "ct")
    except (RuntimeError, ValueError, OSError, EOFError) as exc:
        raise ValueError(f"Could not read CT NIfTI '{image_path}': {exc}") from exc
    try:
        mask, src_m = read_image_robust(mask_path, "mask")
    except (RuntimeError, ValueError, OSError, EOFError) as exc:
        raise ValueError(f"Could not read aorta-mask NIfTI '{mask_path}': {exc}") from exc
    if src_i != "sitk" or src_m != "sitk":
        print(f"[{cid}] image IO fallback ct={src_i}  mask={src_m}")
    if image.GetSize() != mask.GetSize():
        raise ValueError(
            "Image and aorta mask have different sizes: "
            f"image {tuple(image.GetSize())} vs mask {tuple(mask.GetSize())}. "
            "They must share the same voxel grid."
        )
    _log_header_shear(image_path, cid)
    image, mask = _orthonormalize_pair(image, mask, cid)

    ct = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    aorta = sitk.GetArrayFromImage(mask) > 0
    if not np.any(aorta):
        raise ValueError(f"Aorta mask is empty: {mask_path}")

    spacing_xyz = np.array(image.GetSpacing(), dtype=np.float64)
    spacing_zyx = spacing_xyz[::-1].copy()
    return VolumePair(
        image=image,
        mask=mask,
        ct=ct,
        aorta=aorta,
        spacing_zyx=spacing_zyx,
        spacing_xyz=spacing_xyz,
        case_id=cid,
        image_path=image_path,
        mask_path=mask_path,
    )


def index_xyz_to_mm(image: sitk.Image, index_xyz: Sequence[float]) -> np.ndarray:
    """Convert a (possibly fractional) x,y,z index to LPS millimetres."""
    coords = [float(c) for c in index_xyz]
    if all(float(c).is_integer() for c in coords):
        point = image.TransformIndexToPhysicalPoint(tuple(int(round(c)) for c in coords))
    else:
        point = image.TransformContinuousIndexToPhysicalPoint(tuple(coords))
    return np.asarray(point, dtype=np.float64)


def mm_to_index_xyz(image: sitk.Image, point_xyz_mm: Sequence[float]) -> np.ndarray:
    """LPS millimetres → continuous SimpleITK (x, y, z) index."""
    idx = image.TransformPhysicalPointToContinuousIndex(
        tuple(float(c) for c in point_xyz_mm)
    )
    return np.asarray(idx, dtype=np.float64)


def np_zyx_to_mm(image: sitk.Image, z: float, y: float, x: float) -> np.ndarray:
    """Convert a numpy (z, y, x) location to LPS millimetres."""
    return index_xyz_to_mm(image, (x, y, z))


def points_zyx_to_mm(image: sitk.Image, points_zyx: np.ndarray) -> np.ndarray:
    """Vectorised conversion of (N, 3) numpy (z,y,x) points to LPS mm."""
    pts = np.atleast_2d(np.asarray(points_zyx, dtype=np.float64))
    out = np.empty((pts.shape[0], 3), dtype=np.float64)
    for i, (z, y, x) in enumerate(pts):
        out[i] = np_zyx_to_mm(image, z, y, x)
    return out


def bbox_slices(
    mask: np.ndarray,
    margin_voxels: Sequence[int],
) -> tuple[slice, slice, slice]:
    """Axis-aligned bounding box of ``mask`` expanded by per-axis voxel margins."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute bbox of an empty mask")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    margin = np.asarray(margin_voxels, dtype=int)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def crop_pair(
    vol: VolumePair, extra_mm: float
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice], np.ndarray]:
    """Crop CT and aorta to the aorta bbox plus ``extra_mm``.

    Returns ``(ct_c, aorta_c, slices, origin_zyx)`` where ``origin_zyx`` is the
    crop's (z, y, x) offset in the full volume (used to restore indices).
    """
    margin = [max(1, int(np.ceil(extra_mm / s))) for s in vol.spacing_zyx]
    sl = bbox_slices(vol.aorta, margin)
    origin = np.array([s.start for s in sl], dtype=np.int32)
    return vol.ct[sl], vol.aorta[sl], sl, origin
