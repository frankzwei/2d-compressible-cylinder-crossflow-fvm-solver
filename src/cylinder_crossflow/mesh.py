from argparse import HelpFormatter, Namespace
from dataclasses import InitVar, dataclass, field
from math import pi
from numpy.typing import NDArray

import argparse
import gmsh
import math
import numpy as np


@dataclass
class MeshParameters:

    x_min: float = -20
    x_max: float = 50
    y_max: float = 15

    cylinder_radius: float = 0.8
    transition_half_width: float = field(init=False)
    smoothing_half_width: float = field(init=False)

    num_theta_cells: int = 120
    num_prism_cells: int = 24
    num_transition_cells: int = 12
    num_left_cells: int = 24
    num_right_cells: int = 96
    num_top_cells: int = 24
    num_bottom_cells: int = 24

    prism_first_cell_height: float = 1e-3
    prism_growth: float = 1.2
    transition_growth: float = 1
    left_growth: float = 1.115
    right_growth: float = 1.016
    vertical_growth: float = 1.067

    transition_square_cylinder_ratio: InitVar[float] = 4
    smoothing_square_cylinder_ratio: InitVar[float] = 6.5
    max_smoothing_iterations: int = 200
    smoothing_relaxation: float = 0.62

    def __post_init__(self, transition_square_cylinder_ratio: float, smoothing_square_cylinder_ratio: float):
        self.transition_half_width = transition_square_cylinder_ratio * self.cylinder_radius
        self.smoothing_half_width = smoothing_square_cylinder_ratio * self.cylinder_radius

    @property
    def y_min(self) -> float:
        return -self.y_max

    @property
    def prism_thickness(self) -> float:
        return self.prism_first_cell_height * (self.prism_growth ** self.num_prism_cells - 1) / (self.prism_growth - 1)

    @property
    def ogrid_radius(self) -> float:
        return self.cylinder_radius + self.prism_thickness

@dataclass
class GeometryBuilder:

    _points: dict[tuple[float, float], int] = field(default_factory=dict, init=False)
    _lines: dict[tuple[int, int], tuple[int, int, int]] = field(default_factory=dict, init=False)
    _arcs: dict[tuple[int, int, int], tuple[int, int, int]] = field(default_factory=dict, init=False)
    _curve_constraints: dict[int, tuple[int, float]] = field(default_factory=dict, init=False)
    _surface_constraints: dict[int, list[int]] = field(default_factory=dict, init=False)

    @staticmethod
    def _coordinate_key(x: float, y: float) -> tuple[float, float]:
        return round(x, 13), round(y, 13)

    def point(self, x: float, y: float) -> int:
        key: tuple[float, float] = self._coordinate_key(x, y)
        if key not in self._points:
            self._points[key] = gmsh.model.geo.add_point(x, y, 0)

        return self._points[key]

    def line(self, start: int, end: int) -> int:
        key: tuple[int, int] = (min(start, end), max(start, end))
        if key not in self._lines:
            tag: int = gmsh.model.geo.add_line(start, end)
            self._lines[key] = tag, start, end
            return tag

        tag, stored_start, stored_end = self._lines[key]
        return tag if (start, end) == (stored_start, stored_end) else -tag

    def arc(self, start: int, end: int, center: int) -> int:
        key: tuple[int, int, int] = (start, end, center)
        if key not in self._arcs:
            tag: int = gmsh.model.geo.add_circle_arc(start, center, end)
            self._arcs[key] = tag, start, end
            return tag

        tag, stored_start, stored_end = self._arcs[key]
        return tag if (start, end) == (stored_start, stored_end) else -tag

    def set_transfinite_curve(self, tag: int, num_cells: int, progression: float = 1):
        tag = abs(tag)
        stored: tuple[int, float] | None = self._curve_constraints.get(tag)
        if stored is None:
            self._curve_constraints[tag] = num_cells, progression
            return

        if stored != (num_cells, progression):
            raise ValueError(
                f'Attempted to change an existing curve constraint for element {tag} from '
                f'{stored} to ({num_cells}, {progression})'
            )

    def set_transfinite_surface(self, tag: int, corner_tags: list[int] = []):
        tag = abs(tag)
        stored: list[int] = self._surface_constraints.get(tag, [])
        if not stored:
            self._surface_constraints[tag] = corner_tags
            return

        if stored != corner_tags:
            raise ValueError(
                f'Attempted to change an existing surface constraint\'s corner tags for element {tag}'
            )

    def apply_constraints(self):
        for tag, (num_cells, progression) in self._curve_constraints.items():
            gmsh.model.mesh.set_transfinite_curve(tag, num_cells + 1, 'Progression', progression)

        for tag, corner_tags in self._surface_constraints.items():
            gmsh.model.mesh.set_transfinite_surface(tag, cornerTags=corner_tags)
            gmsh.model.mesh.set_recombine(2, tag)

def parse_cli_args(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=lambda prog: HelpFormatter(prog, width=120)
    )

    parser.add_argument('--show-gui', dest='show_gui', action='store_true')

    return parser.parse_args(argv)

def _draw_circular_arcs(
    geometry: GeometryBuilder,
    radius: float,
    center: int,
    num_cells: int
) -> tuple[list[int], list[int]]:
    pts: list[int] = []
    for i in range(4):
        theta: float = 0.25 * pi + 0.5 * pi * i
        x: float = radius * math.cos(theta)
        y: float = radius * math.sin(theta)
        pts.append(geometry.point(x, y))

    arcs: list[int] = []
    for i, start in enumerate(pts):
        end: int = pts[(i + 1) % len(pts)]
        tag: int = geometry.arc(start, end, center)
        arcs.append(tag)
        geometry.set_transfinite_curve(tag, num_cells)

    return pts, arcs

def _draw_rectangular_lines(
    geometry: GeometryBuilder,
    vertices: list[tuple[float, float]],
    num_cells: int,
    progression: float = 1
) -> tuple[list[int], list[int]]:
    pts: list[int] = [geometry.point(x, y) for x, y in vertices]
    lines: list[int] = []
    for i, start in enumerate(pts):
        end: int = pts[(i + 1) % len(pts)]
        tag: int = geometry.line(start, end)
        lines.append(tag)
        geometry.set_transfinite_curve(tag, num_cells, progression)

    return pts, lines

def _draw_lines(
    geometry: GeometryBuilder,
    start_pts: list[int],
    end_pts: list[int],
    num_cells: int,
    progression: float = 1
) -> list[int]:
    if len(start_pts) != len(end_pts):
        raise ValueError(
            f'Attempted to draw lines between {len(start_pts)} starting points and {len(end_pts)} ending points. '
            f'The two lengths must be the same.'
        )

    lines: list[int] = []
    for start, end in zip(start_pts, end_pts):
        tag: int = geometry.line(start, end)
        lines.append(tag)
        geometry.set_transfinite_curve(tag, num_cells, progression)

    return lines

def _form_sector_surface(
    geometry: GeometryBuilder,
    inner_curves: list[int],
    outer_curves: list[int],
    radial_lines: list[int]
) -> list[int]:
    surfaces: list[int] = []
    for i, (minor_radial, outer_curve, inner_curve) in enumerate(zip(radial_lines, outer_curves, inner_curves)):
        major_radial: int = radial_lines[(i + 1) % len(radial_lines)]
        loop: int = gmsh.model.geo.add_curve_loop([minor_radial, outer_curve, -major_radial, -inner_curve])
        surface: int = gmsh.model.geo.add_plane_surface([loop])
        surfaces.append(surface)

        # Set transfinite surface, corner tags are implicitly determined for a shape with 4 corners
        geometry.set_transfinite_surface(surface)

    return surfaces

def _smooth_interfaces(params: MeshParameters) -> int:
    if params.max_smoothing_iterations == 0:
        return 0

    node_tags, xyz_coords, _ = gmsh.model.mesh.get_nodes()
    xy_coords: NDArray[np.float64] = np.reshape(xyz_coords, (-1, 3))[:, :-1].copy()
    tag_to_index: dict[int, int] = {int(tag): i for i, tag in enumerate(node_tags)}

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

    # Convert into a single array and reshape with columns taken two at a time. Each row represents adjacent nodes in
    # counter-clockwise direction. Note that the indices of each node are stored, not the physical node tags themselves
    quads: NDArray[np.int64] = np.vstack(quad_indices)
    edges: NDArray[np.int64] = np.vstack((
        quads[:, [0, 1]],
        quads[:, [1, 2]],
        quads[:, [2, 3]],
        quads[:, [3, 0]]
    ))

    # Each interior edge has contributions from both adjacent quadrilaterals or nodes. Normalize the direction and
    # retain each physical edge only once so ``neighbor_counts`` represent unique neighboring node counts.
    edges: NDArray[np.int64] = np.unique(np.sort(edges, axis=1), axis=0)

    # Create 1D node adjacency arrays. For the same index, the sources and destinations element represent two nodes
    # that are connected to each other by an edge. Note that node neighbors are double counted since node X neighbors
    # node Y, and node Y also neighbors node X.
    sources: NDArray[np.int64] = np.hstack((edges[:, 0], edges[:, 1]))
    destinations: NDArray[np.int64] = np.hstack((edges[:, 1], edges[:, 0]))

    # Count number of occurrences for each node in the entire mesh based off their array index. This stores how many
    # neighbors each node has. Nodes positioned on an external boundary should only have three neighbors.
    neighbor_counts: NDArray[np.int64] = np.bincount(sources, minlength=len(node_tags)).astype(np.int64)

    # Compute distance of all nodes from the origin, used to classify static nodes inside prism layers
    original_xy: NDArray[np.float64] = xy_coords.copy()
    xy_coords_copy: NDArray[np.float64] = original_xy.copy()
    radius: NDArray[np.float64] = np.linalg.norm(original_xy, axis=1)
    max_coordinate: NDArray[np.float64] = np.max(np.abs(original_xy), axis=1)

    # Weight applied to each node adjustment. A value of 0 negates all node displacements whereas a value of 1 permits
    # maximum node displacement as allowed by the smoothing relaxation parameter
    smoothing_weights: NDArray[np.float64] = np.clip(
        (params.smoothing_half_width - max_coordinate) / (params.smoothing_half_width - params.transition_half_width),
        0,
        1
    )

    smoothing_weights[max_coordinate <= params.transition_half_width] = 1
    smoothing_weights[radius <= params.ogrid_radius * (1 + 1e-6)] = 0
    smoothing_weights[neighbor_counts == 0] = 0

    boundaries_mask: NDArray[np.bool] = (
        np.isclose(xy_coords[:, 0], params.x_min) |
        np.isclose(xy_coords[:, 0], params.x_max) |
        np.isclose(xy_coords[:, 1], params.y_min) |
        np.isclose(xy_coords[:, 1], params.y_max)
    )
    smoothing_weights[boundaries_mask] = 0

    for iteration in range(params.max_smoothing_iterations):
        neighbor_sums: NDArray[np.float64] = np.zeros_like(xy_coords_copy, dtype=np.float64)
        np.add.at(neighbor_sums, sources, xy_coords_copy[destinations])

        connected: NDArray[np.bool] = neighbor_counts > 0
        neighbor_average: NDArray[np.float64] = xy_coords_copy.copy()
        neighbor_average[connected] = neighbor_sums[connected] / neighbor_counts[connected, None]

        smoothing_targets: NDArray[np.float64] = (
            smoothing_weights[:, None] * neighbor_average + (1 - smoothing_weights[:, None]) * original_xy
        )

        displacements: NDArray[np.float64] = params.smoothing_relaxation * (smoothing_targets - xy_coords_copy)
        xy_coords_copy += displacements

        if np.max(np.linalg.norm(displacements, axis=1)) <= 1e-10:
            iteration += 1
            break

    for i, tag in enumerate(node_tags):
        coords: list[float] = [float(xy_coords_copy[i, 0]), float(xy_coords_copy[i, 1]), 0]
        gmsh.model.mesh.set_node(tag, coords, [])

    return iteration

def _ignore_construction(*args: tuple[int, list[int]]):
    dim_tags: list[tuple[int, int]] = [(dim, tag) for dim, tags in args for tag in tags]
    gmsh.model.set_visibility(dim_tags, False)

def mesh_domain(params: MeshParameters, display: bool = False):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add('structured_ogrid_mesh')

    geometry: GeometryBuilder = GeometryBuilder()
    origin: int = geometry.point(0, 0)
    sector_theta_cells: int = round(params.num_theta_cells / 4)

    cylinder_pts, cylinder_arcs = _draw_circular_arcs(geometry, params.cylinder_radius, origin, sector_theta_cells)
    ogrid_pts, ogrid_arcs = _draw_circular_arcs(geometry, params.ogrid_radius, origin, sector_theta_cells)

    transition_vertices: list[tuple[float, float]] = [
        (params.transition_half_width, params.transition_half_width),
        (-params.transition_half_width, params.transition_half_width),
        (-params.transition_half_width, -params.transition_half_width),
        (params.transition_half_width, -params.transition_half_width)
    ]

    transition_pts, transition_lines = _draw_rectangular_lines(geometry, transition_vertices, sector_theta_cells)

    # Draw radial lines spanning between cylinder and O-grid circle, and O-grid circle and transition square
    ogrid_radials: list[int] = _draw_lines(
        geometry, cylinder_pts, ogrid_pts, params.num_prism_cells, params.prism_growth
    )

    transition_radials: list[int] = _draw_lines(
        geometry, ogrid_pts, transition_pts, params.num_transition_cells, params.transition_growth
    )

    # Create the prism layer and transition region surface mesh
    ogrid_surfaces: list[int] = _form_sector_surface(geometry, cylinder_arcs, ogrid_arcs, ogrid_radials)
    transition_surfaces: list[int] = _form_sector_surface(geometry, ogrid_arcs, transition_lines, transition_radials)

    # Create the farfield loops. Since the structured mesh generator cannot take in holes, the farfield region is
    # divided into nine quadrants. The center quadrant is the transition square. Horizontal and vertical lines are
    # drawn from each vertex of the transition square to the closest domain outer edge.
    x_coords: list[float] = [params.x_min, -params.transition_half_width, params.transition_half_width, params.x_max]
    y_coords: list[float] = [params.y_min, -params.transition_half_width, params.transition_half_width, params.y_max]
    grid_pts: list[list[int]] = [[geometry.point(x, y) for y in y_coords] for x in x_coords]

    # Set the number of cells in the X and Y directions. Each tuple is defined as the number of cells to "squeeze" or
    # "stack" in the direction. For example, in the X direction, squeeze N cells horizontally.
    x_num_cells: tuple[int, int, int] = (params.num_left_cells, sector_theta_cells, params.num_right_cells)
    y_num_cells: tuple[int, int, int] = (params.num_bottom_cells, sector_theta_cells, params.num_top_cells)

    # Set the growth rate of the cells in the X and Y directions. Each tuple is defined as the growth rate ratio of
    # the cells in the direction. For example, in the X direction, a growth of 1.2 means each successive cell away
    # from the origin (in +X and -X directions) will be 20% larger than the previous cell
    x_growths: tuple[float, float, float] = (1 / params.left_growth, 1, params.right_growth)
    y_growths: tuple[float, float, float] = (1 / params.vertical_growth, 1, params.vertical_growth)

    horizontal_lines: dict[tuple[int, int], int] = {}
    vertical_lines: dict[tuple[int, int], int] = {}
    def _get_horizontal_edge(i: int, j: int) -> int:
        key: tuple[int, int] = (i, j)
        if key not in horizontal_lines:
            tag: int = geometry.line(grid_pts[i][j], grid_pts[i + 1][j])
            horizontal_lines[key] = tag
            geometry.set_transfinite_curve(tag, x_num_cells[i], x_growths[i])

        return horizontal_lines[key]

    def _get_vertical_edge(i: int, j: int) -> int:
        key: tuple[int, int] = (i, j)
        if key not in vertical_lines:
            tag: int = geometry.line(grid_pts[i][j], grid_pts[i][j + 1])
            vertical_lines[key] = tag
            geometry.set_transfinite_curve(tag, y_num_cells[j], y_growths[j])

        return vertical_lines[key]

    quadrant_surfaces: list[int] = []
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                # Skip the center quadrant as this is the transition square
                continue

            # Create the corners of the quadrants
            ll: int = grid_pts[i][j]
            lr: int = grid_pts[i + 1][j]
            ur: int = grid_pts[i + 1][j + 1]
            ul: int = grid_pts[i][j + 1]

            # Get the edges for each quadrant
            left: int = _get_vertical_edge(i, j)
            right: int = _get_vertical_edge(i + 1, j)
            bottom: int = _get_horizontal_edge(i, j)
            top: int = _get_horizontal_edge(i, j + 1)

            # Create the curve loop and surface
            loop: int = gmsh.model.geo.add_curve_loop([bottom, right, -top, -left])
            surface: int = gmsh.model.geo.add_plane_surface([loop])
            geometry.set_transfinite_surface(surface, [ll, lr, ur, ul])
            quadrant_surfaces.append(surface)

    gmsh.model.geo.synchronize()

    for surface in transition_surfaces:
        gmsh.model.mesh.set_smoothing(2, surface, 20)

    geometry.apply_constraints()

    gmsh.option.set_number('Mesh.ColorCarousel', 2)
    gmsh.option.set_number('Mesh.Algorithm', 8)
    gmsh.model.mesh.generate(2)

    quadrant_line_constructions: list[int] = []
    for (i, _), line in vertical_lines.items():
        if i > 0 and i < 3:
            quadrant_line_constructions.append(line)

    for (_, j), line in horizontal_lines.items():
        if j > 0 and j < 3:
            quadrant_line_constructions.append(line)

    quadrant_pt_constructions: list[int] = []
    for i, y_pts in enumerate(grid_pts):
        for j, pt in enumerate(y_pts):
            if (i, j) == (0, 0) or (i, j) == (3, 3) or (i, j) == (0, 3) or (i, j) == (3, 0):
                continue

            quadrant_pt_constructions.append(pt)

    _ignore_construction(
        (1, ogrid_arcs),
        (1, transition_lines),
        (1, ogrid_radials),
        (1, transition_radials),
        (1, quadrant_line_constructions),
        (0, ogrid_pts),
        (0, transition_pts),
        (0, quadrant_pt_constructions),
        (0, cylinder_pts)
    )
    iterations: int = _smooth_interfaces(params)
    print(f'\nRan {iterations} smoothing iterations\n')

    if display:
        gmsh.fltk.run()

    gmsh.finalize()

def main(argv: list[str] | None = None):
    args: Namespace = parse_cli_args(argv)
    params: MeshParameters = MeshParameters()
    mesh_domain(params, args.show_gui)

if __name__ == '__main__':
    main()