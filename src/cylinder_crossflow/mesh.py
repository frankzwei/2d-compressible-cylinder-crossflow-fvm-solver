from dataclasses import dataclass
from math import pi

from cylinder_crossflow.settings import X_MAX, X_MIN, Y_MAX, Y_MIN

import gmsh
import math


@dataclass(frozen=True)
class FarfieldBoundaries:

    x_min: float
    y_min: float
    x_max: float
    y_max: float

DEFAULT_BOUNDARIES: FarfieldBoundaries = FarfieldBoundaries(X_MIN, Y_MIN, X_MAX, Y_MAX)

def check_inequality(small: float, large: float):
    if small >= large:
        raise ValueError(f'Failed the inequality check: minimum value {small} exceeds maximum value {large}')

def check_radius(xc: float, yc: float, farfield_boundaries: FarfieldBoundaries, radius: float):
    max_radius_tuple: tuple[float, float, float, float] = (
        farfield_boundaries.x_max - xc,
        xc - farfield_boundaries.x_min,
        farfield_boundaries.y_max - yc,
        yc - farfield_boundaries.y_min
    )

    max_radius: float = min(max_radius_tuple)
    if radius >= max_radius:
        raise ValueError(
            f'The radius {radius:.2f} must fit completely inside the farfield boundaries'
        )

def generate_radius_points(xc: float, yc: float, radius: float, ogrid_radius: float) -> tuple[list[int], list[int]]:
    """Return a list of tags uniquely associated with the points that make up the cylinder and ogrid radius.

    The points are ordered counter-clockwise, starting at the +X axis. By default, only 8 points are used to
    generate the O-grid mesh.

    :param xc: The X coordinate of the cylinder's center.
    :param yc: The Y coordinate of the cylinder's center.
    :param radius: The cylinder radius.
    :param ogrid_radius: The radius of the o-grid interface.
    :return: A tuple of the list of cylinder and O-grid radius points.
    """
    cylinder_pts: list[int] = []
    ogrid_radius_pts: list[int] = []
    for i in range(8):
        theta = 0.25 * pi * i
        x: float = xc + radius * math.cos(theta)
        y: float = yc + radius * math.sin(theta)

        cylinder_pts.append(gmsh.model.geo.add_point(x, y, 0))

        x: float = xc + ogrid_radius * math.cos(theta)
        y: float = yc + ogrid_radius * math.sin(theta)

        ogrid_radius_pts.append(gmsh.model.geo.add_point(x, y, 0))

    return cylinder_pts, ogrid_radius_pts

def generate_farfield_pts(xc: float, yc: float, farfield_boundaries: FarfieldBoundaries) -> list[int]:
    """Return a list of tags uniquely associated with the points that make up the farfield boundaries.

    The points are ordered counter-clockwise, starting at the +X radius.

    :param xc: The X coordinate of the cylinder's center.
    :param yc: The Y coordinate of the cylinder's center.
    :param farfield_boundaries: The namespace representation of the farfield coordinates.
    :return: A list of tags for the farfield coordinates.
    """
    coordinates: list[tuple[float, float]] = [
        (farfield_boundaries.x_max, yc),
        (farfield_boundaries.x_max, farfield_boundaries.y_max),
        (xc, farfield_boundaries.y_max),
        (farfield_boundaries.x_min, farfield_boundaries.y_max),
        (farfield_boundaries.x_min, yc),
        (farfield_boundaries.x_min, farfield_boundaries.y_min),
        (xc, farfield_boundaries.y_min),
        (farfield_boundaries.x_max, farfield_boundaries.y_min)
    ]

    return [gmsh.model.geo.add_point(x, y, 0) for x, y in coordinates]

def draw_circle_arcs(center_tag: int, tags: list[int]) -> list[int]:
    return [
        gmsh.model.geo.add_circle_arc(
            tag,
            center_tag,
            tags[(i + 1) % 8]
        ) for i, tag in enumerate(tags)
    ]

def draw_farfield_lines(tags: list[int]) -> list[int]:
    return [
        gmsh.model.geo.add_line(
            tag,
            tags[(i + 1) % 8]
        ) for i, tag in enumerate(tags)
    ]

def form_sector_surfaces(inner_tags: list[int], outer_tags: list[int]) -> tuple[list[int], list[int]]:
    """Return a list of tags uniquely associated with a surface for each sector.

    Creates a CAD surface for each of the eight sectors in the domain. Each sector is defined as having an
    inner and outer bounding curve/line. The method first creates additional lines spanning radially outwards
    before creating the surface that connects the associated radial lines and inner/outer lines together. Each
    surface is oriented counter-clockwise and the method returns the sectors ordered in the counter-clockwise
    direction.

    :param inner_tags: The list of tags for the inner curves or lines.
    :param outer_tags: The list of tags for the outer curves or lines.
    :return: The list of surface tags for each sector and line tags for the radial lines in counter-clockwise order.
    """
    def draw_radial_lines() -> list[int]:
        """Return a list of tags uniquely associated with the radial lines that connect respective points along
        the inner and outer curves.

        :raises ValueError: If the number of points differs between the inner and outer curves.
        :return: The list of tags of the radial lines.
        """
        if len(inner_tags) != len(outer_tags):
            raise ValueError(
                f'The number of points in the inner and outer loops do not match, got {len(inner_tags)}'
                f'inner points and {len(outer_tags)} outer points'
            )

        return [
            gmsh.model.geo.add_line(
                inner_tag,
                outer_tag
            ) for inner_tag, outer_tag in zip(inner_tags, outer_tags)
        ]

    radial_lines: list[int] = draw_radial_lines()

    sector_surfaces: list[int] = []
    for i, (inner_tag, outer_tag, radial_tag) in enumerate(zip(inner_tags, outer_tags, radial_lines)):
        loop_tag: int = gmsh.model.geo.add_curve_loop([radial_tag, outer_tag, -radial_lines[(i + 1) % 8], -inner_tag])
        sector_surfaces.append(gmsh.model.geo.add_plane_surface([loop_tag]))

    return sector_surfaces, radial_lines

def set_theta_transfinite_curve(
    n_theta_sectors: int,
    cylinder_arcs: list[int],
    ogrid_transition_arcs: list[int],
    farfield_lines: list[int]
):
    for cylinder_arc, ogrid_arc, farfield_line in zip(cylinder_arcs, ogrid_transition_arcs, farfield_lines):
        gmsh.model.mesh.set_transfinite_curve(cylinder_arc, n_theta_sectors + 1)
        gmsh.model.mesh.set_transfinite_curve(ogrid_arc, n_theta_sectors + 1)
        gmsh.model.mesh.set_transfinite_curve(farfield_line, n_theta_sectors + 1)

def set_radial_transfinite_curve(n_radial_layers: int, curves: list[int], progression: float):
    for curve in curves:
        gmsh.model.mesh.set_transfinite_curve(curve, n_radial_layers + 1, 'Progression', progression)

def mesh_domain(
    radius: float = 0.5,
    ogrid_radius: float = 5,
    ogrid_growth: float = 1.1,
    farfield_growth: float = 1.1,
    farfield_boundaries: FarfieldBoundaries = DEFAULT_BOUNDARIES,
    n_theta_sectors: int = 12,
    n_radial_ogrid: int = 20,
    n_radial_farfield: int = 30,
    filename: str | None = None,
    display: bool = True
):
    gmsh.initialize()
    gmsh.model.add('2d_cylinder_compressible_crossflow')

    # Define center coordinates of circle
    xc: float = 0
    yc: float = 0

    # Check the cylinder and O-grid radius are within the farfield boundaries. This does not check for
    # whether the cylinder and O-grid radius comfortably fit inside the domain boundaries.
    check_radius(xc, yc, farfield_boundaries, radius)
    check_radius(xc, yc, farfield_boundaries, ogrid_radius)

    # Verify the O-grid radius is larger than the cylinder radius
    check_inequality(radius, ogrid_radius)

    center_pt = gmsh.model.geo.add_point(xc, yc, 0)
    cylinder_pts, ogrid_radius_pts = generate_radius_points(xc, yc, radius, ogrid_radius)
    farfield_pts = generate_farfield_pts(xc, yc, farfield_boundaries)

    # Form the connections for cylinder radius, O-grid transition radius, and farfield boundaries.
    cylinder_arcs: list[int] = draw_circle_arcs(center_pt, cylinder_pts)
    ogrid_transition_arcs: list[int] = draw_circle_arcs(center_pt, ogrid_radius_pts)
    farfield_lines: list[int] = draw_farfield_lines(farfield_pts)

    # Form surfaces within each loop across all eight sectors. Each surface's loop starts at the minor radial line and
    # travels in the counter-clockwise direction.
    inner_sector_surfaces, inner_radial_lines = form_sector_surfaces(cylinder_arcs, ogrid_transition_arcs)
    outer_sector_surfaces, outer_radial_lines = form_sector_surfaces(ogrid_transition_arcs, farfield_lines)

    # Synchronize the geometry
    gmsh.model.geo.synchronize()

    # Divide the radial lines and arcs by the respective number of nodes by setting the number of transfinite curves
    # in the mesh object.
    set_theta_transfinite_curve(n_theta_sectors, cylinder_arcs, ogrid_transition_arcs, farfield_lines)
    set_radial_transfinite_curve(n_radial_ogrid, inner_radial_lines, ogrid_growth)
    set_radial_transfinite_curve(n_radial_farfield, outer_radial_lines, farfield_growth)

    surfaces: list[int] = inner_sector_surfaces + outer_sector_surfaces
    for surface in surfaces:
        gmsh.model.mesh.set_transfinite_surface(surface)
        gmsh.model.mesh.set_recombine(2, surface)

    gmsh.model.mesh.generate(2)

    if filename:
        gmsh.write(filename)

    if display:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == '__main__':
    mesh_domain()