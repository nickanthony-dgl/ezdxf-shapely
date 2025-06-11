from collections.abc import Iterable

import shapely
import shapely.geometry as sg
from shapely import ops
from shapely import affinity

__all__ = ["coerce_line_ends", "polygonize", "centralize", "line_merge"]


def coerce_line_ends(geoms: Iterable[sg.LineString], distance: float = 1e-8) -> list[sg.LineString]:
    """
    Coerce nearby line ends to the exact same point.

    :param geoms: iterable of line strings to operate on
    :param distance: maximum distance to move line ends during coercion

    :returns: the line strings with coerced ends (fresh instances)

    TODO merge geometries rather than just adding additional points to one of them
    """

    geoms = list(geoms)
    for i in range(len(geoms)):
        line_1 = geoms[i]
        first_point_1 = sg.Point(line_1.coords[0])  # startpoint
        last_point_1 = sg.Point(line_1.coords[-1])  # endpoint

        for j in range(i + 1, len(geoms)):
            line_2 = geoms[j]
            first_point_2 = sg.Point(line_2.coords[0])
            last_point_2 = sg.Point(line_2.coords[-1])
            distance_first_points = first_point_1.distance(first_point_2)
            distance_mixed_1 = first_point_1.distance(last_point_2)
            distance_mixed_2 = last_point_1.distance(first_point_2)
            dist_last_points = last_point_1.distance(last_point_2)
            if 0 < distance_first_points < distance:
                geoms[j] = sg.LineString([line_1.coords[0]] + line_2.coords[1:])
            if 0 < distance_mixed_1 < distance:
                geoms[j] = sg.LineString(line_2.coords[:-1] + [line_1.coords[0]])
            if 0 < distance_mixed_2 < distance:
                geoms[j] = sg.LineString([line_1.coords[-1]] + line_2.coords[1:])
            if 0 < dist_last_points < distance:
                geoms[j] = sg.LineString(line_2.coords[:-1] + [line_1.coords[-1]])
    return geoms


def polygonize(
    geoms: Iterable[sg.LineString], coercion_distance: float | None = 1e-8, simplify=True
) -> list[sg.Polygon]:
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
    geoms: Iterable[sg.LineString], coerce_ends=True, coercion_distance=1e-8, simplify=True
) -> sg.LineString | sg.MultiLineString:
    """
    Create merged line strings from the given partial line strings.
    Optionally, coerce the line ends before merging and simplify the result after.

    :param geoms: iterable of line strings to operate on
    :param coerce_ends: whether to coerce the line ends before merging
    :param coercion_distance: maximum distance to move line ends during coercion
    :param simplify: whether to simplify the resulting merged strings

    :returns: the merged line string, may be a multi-line-string if the lines have gaps
    """
    merged = shapely.line_merge(geoms)
    if coerce_ends:
        if isinstance(sg.LineString):
            merged = [merged]  # Must be iterable for coerce_line_ends
        merged = coerce_line_ends(merged, coercion_distance)
        merged = shapely.line_merge(geoms)
    if simplify:
        merged = merged.simplify(0)
    return merged


def centralize(geoms: Iterable[sg.base.BaseGeometry] | sg.base.BaseGeometry) -> list[sg.base.BaseGeometry]:
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
