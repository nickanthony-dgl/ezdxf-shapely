from collections.abc import Iterable

import numba
import numba.types as nbt
import numpy as np
import shapely
from shapely import ops
from shapely import affinity
from numpy.typing import NDArray

__all__ = ["coerce_line_ends", "polygonize", "centralize", "line_merge"]


def _is_near(a: NDArray, b: NDArray, dist: float) -> bool:
    """Check if two points are within dist of eachother"""
    return abs(a[0] - b[0]) < dist and abs(a[1] - b[1]) < dist


def _connect_path(a: NDArray, b: NDArray, dist: float) -> NDArray | None:
    """If an end point of B is within distance of an endpoint of A then modify A by connecting B to it"""
    if _is_near(a[0], b[0], dist):
        return np.vstack((a[::-1], b[1:]))  # connect reversed A to B (minus coinciding point)
    if _is_near(a[0], b[-1], dist):
        return np.vstack((b[:-1], a))
    if _is_near(a[-1], b[0], dist):
        return np.vstack((a[:-1], b))
    if _is_near(a[-1], b[-1], dist):
        return np.vstack((a[:-1], b[::-1]))  # connect A to reversed B (minus coinciding point)
    return None


def _build_single_path(lines: list[NDArray], distance: float) -> NDArray:
    path = lines[0]
    del lines[0]

    while True:  # Keep processing until we have an iteration where nothing happens
        shouldDelete: list[int] = []  # This is guaranteed to be sorted
        for j in range(len(lines)):
            b = lines[j]
            result = _connect_path(path, b, distance)
            if result is not None:
                shouldDelete.append(j)
                path = result
        # remove merged lines from future consideration
        if len(shouldDelete) > 0:
            for idx in reversed(shouldDelete):  # Delete in reverse order (so that earlier indices to delete aren't invalidated).
                del lines[idx]
        else:
            # We didn't find anything to connect, we're done
            if _is_near(path[0], path[-1], distance):
                # Check if this path can be closed upon itself
                path[-1] = path[0]
            return path


def _merge_with_tolerance(lines: list[NDArray], distance: float) -> list[NDArray]:
    """
    Each line should be an Nx2 coordinate sequence. Note `lines` will be empty by the end of the function

    Returns a list of merged coordinate sequences
    """
    out = []
    while len(lines) > 0:
        out.append(_build_single_path(lines, distance))
    return out


def coerce_line_ends(geoms: Iterable[shapely.LineString], distance: float = 1e-8) -> list[shapely.LineString]:
    """
    Coerce nearby line ends to the exact same point.

    :param geoms: iterable of line strings to operate on
    :param distance: maximum distance to move line ends during coercion

    :returns: the merged line strings with coerced ends (fresh instances)
    """
    lines = [np.array(l.coords) for l in geoms]
    merged = _merge_with_tolerance(lines, distance)
    return [shapely.LineString(m) for m in merged]


def polygonize(
    geoms: Iterable[shapely.LineString], coercion_distance: float | None = 1e-8, simplify=True
) -> list[shapely.Polygon]:
    """
    Create polygons from the given line strings.
    Optionally, coerce the line ends before polygonization and simplify the result after.

    :param geoms: iterable of line strings to use for polygonization
    :param coercion_distance: If specified the line ends will be coerced before polygonization with this as the maximum distance to move line ends during coercion
    :param simplify: whether to simplify the resulting polygons

    :returns: a list of created polygons
    """
    merged = ops.linemerge(geoms)
    if isinstance(merged, shapely.LineString):
        # The lines were already merged to a single line
        polygons = list(ops.polygonize(merged))
    else:
        assert isinstance(merged, shapely.MultiLineString)
        if coercion_distance is not None:
            merged = coerce_line_ends(list(merged.geoms), coercion_distance)
        polygons = list(ops.polygonize(merged))
    if simplify:
        polygons = [p.simplify(0) for p in polygons]
    return polygons


def line_merge(
    geoms: Iterable[shapely.LineString], coerce_ends=True, coercion_distance=1e-8, simplify=True
) -> shapely.LineString | shapely.MultiLineString:
    """
    Create merged line strings from the given partial line strings.
    Optionally, coerce the line ends before merging and simplify the result after.

    :param geoms: iterable of line strings to operate on
    :param coerce_ends: whether to coerce the line ends before merging
    :param coercion_distance: maximum distance to move line ends during coercion
    :param simplify: whether to simplify the resulting merged strings

    :returns: the merged line string, may be a multi-line-string if the lines have gaps
    """
    merged = ops.linemerge(geoms)
    if coerce_ends:
        if isinstance(merged, shapely.LineString):
            merged = [merged]  # Must be iterable for coerce_line_ends
        else:
            assert isinstance(merged, shapely.MultiLineString)
            merged = list(merged.geoms)
        merged = coerce_line_ends(merged, coercion_distance)
        if len(merged) == 1:
            merged = merged[0]
        else:
            # Convert list of linestrings to shapely multilinestring
            merged = shapely.MultiLineString(merged)
    if simplify:
        merged = merged.simplify(0)
    return merged


def centralize(geoms: Iterable[shapely.Geometry] | shapely.Geometry) -> list[shapely.Geometry]:
    """
    Translate all given geometries so that their centroid is in the origin (0, 0).
    Translation is done for each independently.
    Create multi-geometries in advance to regard them as a whole.

    :param geoms: iterable of geometries or a single geometry to operate on

    :returns: a list of translated line strings, also if just a single geometry was passed
    """
    if not isinstance(geoms, Iterable):
        geoms = [geoms]
    return [affinity.translate(l, -l.centroid.x, -l.centroid.y) for l in geoms]
