from argparse import HelpFormatter, Namespace
from dataclasses import dataclass, field
from itertools import chain
from math import pi
from numpy.typing import NDArray

import argparse
import gmsh
import math

import numpy as np


@dataclass(frozen=True)
class MeshParameters:

    x_min: float = -20
    x_max: float = 50
    y_max: float = 15

    cylinder_radius: float = 1
    local_cell_size: float = 0.1
    transition_cell_size: float = 0.3
    base_cell_size: float = 0.5

    prism_first_cell_height: float = 1e-3
    prism_layers: int = 30
    prism_growth: float = 1.2

    transition_growth: float = 1
    farfield_left_growth: float = 1.01
    farfield_right_growth: float = 1.01
    farfield_vertical_growth: float = 1

    transition_square_scale: float = 0.5
    max_smoothing_iterations: int = 500
    smoothing_relaxation: float = 0.8

    @property
    def y_min(self) -> float:
        return -self.y_max

    @property
    def prism_thickness(self) -> float:
        return self.prism_first_cell_height * (self.prism_growth ** self.prism_layers - 1) / (self.prism_growth - 1)

    @property
    def ogrid_radius(self) -> float:
        return self.cylinder_radius + self.prism_thickness

    @property
    def transition_half_width(self) -> float:
        return self.transition_square_scale * min((
            0.5 * (self.ogrid_radius + self.x_max),
            0.5 * (self.ogrid_radius - self.x_min),
            0.5 * (self.ogrid_radius + self.y_max)
        ))

    @property
    def num_theta_cells_per_sector(self) -> int:
        """Compute the number of cells in the tangential direction for each sector.

        The annular sector between the cylinder and O-grid radius, and the slice between the O-grid radius and
        transition square, must be meshed with the same number of cells for a structured mesh. The method computes
        the nearest integer number of cells that satisfies the local size constraint at the cylinder. The actual
        cell size will likely not be equal to the local size input by the user.

        :return: The number of cells in the tangential direction to mesh with.
        """
        return round(0.5 * pi * self.cylinder_radius / self.local_cell_size)

    @property
    def num_radial_transition_cells(self) -> int:
        """Compute the number of cells in the radial direction to fit in the transition region.

        Determines the number of cells to "stack" between the O-grid radius and transition square. The method computes
        the nearest integer number of cells that satisfies the local transition cell size constraint as input by the
        user. The actual cell size will likely not be equal to the original user input size.

        :return: The number of cells in the radial direction to mesh with in the transition region.
        """
        return round((self.transition_half_width - self.ogrid_radius) / self.transition_cell_size)


@dataclass(frozen=True)
class GeometryBuilder:

    _points: dict[tuple[float, float], int] = field(default_factory=dict, init=False)
    _curves: dict[tuple[int, int, int], tuple[int, int, int]] = field(default_factory=dict, init=False)
    _lines: dict[tuple[int, int], tuple[int, int, int]] = field(default_factory=dict, init=False)
    _curve_constraints: dict[int, tuple[int, str, float]] = field(default_factory=dict, init=False)

    @staticmethod
    def _key(x: float, y: float) -> tuple[float, float]:
        return (round(x, 13), round(y, 13))

    def point(self, x: float, y: float) -> int:
        key: tuple[float, float] = self._key(x, y)
        if (tag := self._points.get(key)) is None:
            tag = gmsh.model.geo.add_point(x, y, 0)
            self._points[key] = tag

        return tag

    def curve(self, start: int, end: int, center: int) -> int:
        key: tuple[int, int, int] = (min(start, end), max(start, end), center)
        if key not in self._curves:
            tag: int = gmsh.model.geo.add_circle_arc(start, center, end)
            self._curves[key] = tag, start, end
            return tag

        tag, stored_start, stored_end = self._curves[key]
        return tag if (stored_start, stored_end) == (start, end) else -tag

    def line(self, start: int, end: int) -> int:
        key: tuple[int, int] = (min(start, end), max(start, end))
        if key not in self._lines:
            tag: int = gmsh.model.geo.add_line(start, end)
            self._lines[key] = tag, start, end
            return tag

        tag, stored_start, stored_end = self._lines[key]
        return tag if (stored_start, stored_end) == (start, end) else -tag

    def add_curve_constraint(self, signed_tag: int, num_cells: int, mesh_type: str = 'Progression', growth: float = 1):
        tag: int = abs(signed_tag)
        stored: tuple[int, str, float] | None = self._curve_constraints.get(tag)
        if stored is None:
            self._curve_constraints[tag] = num_cells + 1, mesh_type, growth
            return

        if stored != (num_cells + 1, mesh_type, growth):
            raise ValueError(f'Attempted to change an existing curve constraint for element {tag}')

    def apply_curve_constraints(self):
        for tag, (nodes, progression, growth) in self._curve_constraints.items():
            gmsh.model.mesh.set_transfinite_curve(tag, nodes, progression, growth)

def parse_cli_args(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=lambda prog: HelpFormatter(prog, width=120)
    )

    parser.add_argument('--show-gui', dest='show_gui', action='store_true')

    return parser.parse_args(argv)

def _draw_transition_lines(params: MeshParameters, geometry: GeometryBuilder) -> tuple[list[int], list[int]]:
    vertices: list[tuple[float, float]] = [
        (params.transition_half_width, params.transition_half_width),
        (-params.transition_half_width, params.transition_half_width),
        (-params.transition_half_width, -params.transition_half_width),
        (params.transition_half_width, -params.transition_half_width)
    ]

    points: list[int] = [geometry.point(x, y) for x, y in vertices]
    lines: list[int] = []
    for i, start in enumerate(points):
        j = (i + 1) % len(points)
        lines.append(line := geometry.line(start, points[j]))
        geometry.add_curve_constraint(line, params.num_theta_cells_per_sector)

    return points, lines

def _draw_farfield_lines(params: MeshParameters, geometry: GeometryBuilder) -> tuple[list[int], list[int]]:
    vertices: list[tuple[float, float]] = [
        (params.x_max, params.y_max),
        (params.x_min, params.y_max),
        (params.x_min, params.y_min),
        (params.x_max, params.y_min)
    ]

    points: list[int] = [geometry.point(x, y) for x, y in vertices]
    lines: list[int] = []
    for i, start in enumerate(points):
        j = (i + 1) % len(points)
        lines.append(line := geometry.line(start, points[j]))
        geometry.add_curve_constraint(line, round(math.dist(vertices[i], vertices[j]) / params.base_cell_size))

    return points, lines

def _draw_circle_arcs(
    geometry: GeometryBuilder,
    radius: float,
    center: int,
    num_cells: int
) -> tuple[list[int], list[int]]:
    points: list[int] = []
    for i in range(4):
        theta: float = 0.25 * pi + 0.5 * pi * i
        points.append(geometry.point(radius * math.cos(theta), radius * math.sin(theta)))

    arcs: list[int] = []
    for i, start in enumerate(points):
        end: int = points[(i + 1) % len(points)]
        arcs.append(arc := geometry.curve(start, end, center))
        geometry.add_curve_constraint(arc, num_cells)

    return points, arcs

def _draw_radial_lines(
    geometry: GeometryBuilder,
    inner_pts: list[int],
    outer_pts: list[int],
    num_cells: int,
    growth: float = 1
) -> list[int]:
    lines: list[int] = []
    for start, end in zip(inner_pts, outer_pts):
        lines.append(line := geometry.line(start, end))
        geometry.add_curve_constraint(line, num_cells, 'Progression', growth)

    return lines

def _form_sector_surfaces(inner_curves: list[int], outer_curves: list[int], radial_lines: list[int]) -> list[int]:
    surfaces: list[int] = []
    for i, (inner_curve, outer_curve, radial_line) in enumerate(zip(inner_curves, outer_curves, radial_lines)):
        loop: int = gmsh.model.geo.add_curve_loop([
            radial_line,
            outer_curve,
            -radial_lines[(i + 1) % len(radial_lines)],
            -inner_curve
        ])

        surfaces.append(surface := gmsh.model.geo.add_plane_surface([loop]))
        gmsh.model.geo.mesh.set_transfinite_surface(surface)
        gmsh.model.geo.mesh.set_recombine(2, surface)

    return surfaces

def _form_farfield_quadrant_surfaces(
    params: MeshParameters,
    geometry: GeometryBuilder,
    farfield_pts: list[int]
) -> tuple[list[int], list[int], list[int], list[int]]:
    x_coordinates: list[float] = [
        params.x_min,
        -params.transition_half_width,
        params.transition_half_width,
        params.x_max
    ]

    y_coordinates: list[float] = [
        params.y_min,
        -params.transition_half_width,
        params.transition_half_width,
        params.y_max
    ]

    grid_pts: list[list[int]] = [[geometry.point(x, y) for y in y_coordinates] for x in x_coordinates]
    x_growths: list[float] = [1 / params.farfield_left_growth, 1, params.farfield_right_growth]
    y_growths: list[float] = [1 / params.farfield_vertical_growth, 1, params.farfield_vertical_growth]

    horizontal_lines: dict[tuple[int, int], int] = {}
    vertical_lines: dict[tuple[int, int], int] = {}

    def _get_horizontal_edge(x: int, y: int) -> tuple[int, float]:
        key: tuple[int, int] = (x, y)
        if (tag := horizontal_lines.get(key)) is None:
            tag = geometry.line(grid_pts[x][y], grid_pts[x + 1][y])
            horizontal_lines[key] = tag

        return tag, x_coordinates[x + 1] - x_coordinates[x]

    def _get_vertical_edge(x: int, y: int) -> tuple[int, float]:
        key: tuple[int, int] = (x, y)
        if (tag := vertical_lines.get(key)) is None:
            tag = geometry.line(grid_pts[x][y], grid_pts[x][y + 1])
            vertical_lines[key] = tag

        return tag, y_coordinates[y + 1] - y_coordinates[y]

    def _constrain_quadrant_edges(x: int, y: int) -> tuple[int, int, int, int]:
        left, left_length = _get_vertical_edge(x, y)
        right, _ = _get_vertical_edge(x + 1, y)
        top, _ = _get_horizontal_edge(x, y + 1)
        bottom, bottom_length = _get_horizontal_edge(x, y)

        if (x, y) == (0, 1) or (x, y) == (2, 1):
            # Corresponds to left and right center quadrants. The vertical edges must match the number of theta cells
            # per sector. The top and bottom edges are computed based off the farfield cell size
            geometry.add_curve_constraint(left, params.num_theta_cells_per_sector)
            geometry.add_curve_constraint(right, params.num_theta_cells_per_sector)

            num_cells: int = round(bottom_length / params.base_cell_size)
            geometry.add_curve_constraint(bottom, num_cells, 'Progression', x_growths[x])
            geometry.add_curve_constraint(top, num_cells, 'Progression', x_growths[x])
        elif (x, y) == (1, 0) or (x, y) == (1, 2):
            # Corresponds to bottom and top center quadrants. The top and bottom edges must match the number of theta
            # cells per sector. The left and right edges are computed based off the farfield cell size
            geometry.add_curve_constraint(bottom, params.num_theta_cells_per_sector)
            geometry.add_curve_constraint(top, params.num_theta_cells_per_sector)

            num_cells: int = round(left_length / params.base_cell_size)
            geometry.add_curve_constraint(left, num_cells, 'Progression', y_growths[y])
            geometry.add_curve_constraint(right, num_cells, 'Progression', y_growths[y])
        else:
            # Corresponds to corner quadrants. Farfield cell size is used to compute the number of transfinite nodes
            # for all edges
            left_cells: int = round(left_length / params.base_cell_size)
            bottom_cells: int = round(bottom_length / params.base_cell_size)
            geometry.add_curve_constraint(left, left_cells, 'Progression', y_growths[y])
            geometry.add_curve_constraint(right, left_cells, 'Progression', y_growths[y])
            geometry.add_curve_constraint(bottom, bottom_cells, 'Progression', x_growths[x])
            geometry.add_curve_constraint(top, bottom_cells, 'Progression', x_growths[x])

        return left, right, bottom, top

    farfield_surfaces: list[int] = []
    for x in range(3):
        for y in range(3):
            if x == 1 and y == 1:
                # Skip the center quadrant as that corresponds to the transition square region
                continue

            # Add curve constraints to farfield quadrants. For center quadrants, the edge opposite to the transition
            # square edge must match the number of transfinite nodes. For instance, with quadrant (0, 1), the left
            # edge will have the same number of transfinite nodes as the right edge (transition square side). The
            # remaining edges will compute the number of nodes based off the farfield cell size.
            left, right, bottom, top = _constrain_quadrant_edges(x, y)
            loop: int = gmsh.model.geo.add_curve_loop([right, -top, -left, bottom])
            farfield_surfaces.append(surface := gmsh.model.geo.add_plane_surface([loop]))

            # Mark the quadrant as a transfinite surface to explicitly create a structured mesh. Set the corner tags
            # based off the grid points explicitly. This step is not required but is good to have.
            pt_ll: int = grid_pts[x][y]
            pt_lr: int = grid_pts[x + 1][y]
            pt_ur: int = grid_pts[x + 1][y + 1]
            pt_ul: int = grid_pts[x][y + 1]
            gmsh.model.geo.mesh.set_transfinite_surface(surface, cornerTags=[pt_ll, pt_lr, pt_ur, pt_ul])
            gmsh.model.geo.mesh.set_recombine(2, surface)

    intermediate_pts: list[list[int]] = [[y for y in row if y not in farfield_pts] for row in grid_pts]
    return (
        farfield_surfaces,
        list(horizontal_lines.values()),
        list(vertical_lines.values()),
        list(chain.from_iterable(intermediate_pts))
    )

def _smooth_interface(params: MeshParameters):
    if params.max_smoothing_iterations == 0:
        return

    node_tags, xyz_coords, _ = gmsh.model.mesh.get_nodes()
    xy_coords: NDArray[np.float64] = np.reshape(xyz_coords, (-1, 3))[:, :-1]
    tag_to_index: dict[int, int] = {tag: i for i, tag in enumerate(node_tags)}

    quad_indices: list[NDArray[np.int64]] = []
    element_types, _, element_node_tags = gmsh.model.mesh.get_elements(2)
    for element_type, connectivity in zip(element_types, element_node_tags):
        name, _, _, num_nodes, _, _ = gmsh.model.mesh.get_element_properties(element_type)

        # Skip element shapes that are not quadrilaterals. The element name differs from the family name (e.g., Point)
        # in that 1st order quadrangle elements will have an element name of 'Quadrilateral 4'. 2nd order elements
        # will have a name of 'Quadrilateral 8'
        if not name.startswith('Quadrilateral'):
            continue

        # Remove nodes for higher-order elements, only keep the corner nodes in counter-clockwise order. This
        # guarantees the array will only have four columns corresponding to each corner node
        element_nodes: NDArray[np.int64] = np.reshape(connectivity, (-1, num_nodes))[:, :4]

        # Element nodes is an (N, 4) array with element index as the row and corresponding node tag as the column. The
        # nodes are ordered in VTK/ISO standard (counter-clockwise rotation starting from bottom left as 0). Convert
        # the array from node tags to array indices and store in the quadrilateral index list.
        indices: NDArray[np.int64] = np.fromiter(
            (tag_to_index[int(tag)] for tag in element_nodes.ravel()),
            dtype=np.int64
        )

        quad_indices.append(indices.reshape(-1, 4))

    for iteration in range(params.max_smoothing_iterations):
        pass

def _ignore_constructions(*args: tuple[int, list[int]]):
    dim_tags: list[tuple[int, int]] = []
    for dim, tags in args:
        for tag in tags:
            dim_tags.append((dim, tag))

    gmsh.model.set_visibility(dim_tags, False)

def mesh_domain(params: MeshParameters, display: bool = False):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add('structured_ogrid_mesh')

    geometry: GeometryBuilder = GeometryBuilder()
    center: int = geometry.point(0, 0)

    cylinder_pts, cylinder_arcs = _draw_circle_arcs(
        geometry,
        params.cylinder_radius,
        center,
        params.num_theta_cells_per_sector
    )

    ogrid_pts, ogrid_arcs = _draw_circle_arcs(
        geometry,
        params.ogrid_radius,
        center,
        params.num_theta_cells_per_sector
    )

    transition_pts, transition_lines = _draw_transition_lines(params, geometry)
    farfield_pts, farfield_lines = _draw_farfield_lines(params, geometry)

    # Draw radial lines angled at 45 degrees from the x-axis, separate lines are created for the O-grid circle and
    # transition square
    ogrid_radial_lines: list[int] = _draw_radial_lines(
        geometry,
        cylinder_pts,
        ogrid_pts,
        params.prism_layers,
        params.prism_growth
    )

    transition_radial_lines: list[int] = _draw_radial_lines(
        geometry,
        ogrid_pts,
        transition_pts,
        params.num_radial_transition_cells,
    )

    # Create the surface for each 45 degree sector in the O-grid circle and transition square
    ogrid_surfaces: list[int] = _form_sector_surfaces(cylinder_arcs, ogrid_arcs, ogrid_radial_lines)
    transition_surfaces: list[int] = _form_sector_surfaces(ogrid_arcs, transition_lines, transition_radial_lines)
    farfield_surfaces, horizontals, verticals, intermediate_pts = _form_farfield_quadrant_surfaces(
        params,
        geometry,
        farfield_pts
    )

    gmsh.model.geo.synchronize()

    geometry.apply_curve_constraints()
    for surface in transition_surfaces:
        gmsh.model.mesh.set_smoothing(2, surface, 20)

    # Add physical groups for the different boundaries and surfaces
    surfaces: list[int] = ogrid_surfaces + transition_surfaces + farfield_surfaces
    gmsh.model.add_physical_group(2, surfaces, name='Fluid')
    gmsh.model.add_physical_group(1, cylinder_arcs, name='Cylinder')
    gmsh.model.add_physical_group(1, farfield_lines, name='Farfield')

    gmsh.option.set_number('Mesh.ColorCarousel', 2)
    gmsh.option.set_number('Mesh.Algorithm', 8)
    gmsh.model.mesh.generate(2)

    # Smooth the connectivity in the transition square. Laplacian smoothing is applied, where an arbitrary node is
    # adjusted based off the average position of its neighboring nodes. Physically, this resembles an energy
    # conservation where node connections are replaced with elastic springs.
    _smooth_interface(params)
    _ignore_constructions(
        (1, ogrid_arcs),
        (1, transition_lines),
        (1, ogrid_radial_lines),
        (1, transition_radial_lines),
        (1, horizontals),
        (1, verticals),
        (0, ogrid_pts),
        (0, transition_pts),
        (0, intermediate_pts)
    )

    if display:
        gmsh.fltk.run()

    gmsh.finalize()

def main(argv: list[str] | None = None):
    args: Namespace = parse_cli_args(argv)
    params: MeshParameters = MeshParameters()
    mesh_domain(params, args.show_gui)

if __name__ == '__main__':
    main()