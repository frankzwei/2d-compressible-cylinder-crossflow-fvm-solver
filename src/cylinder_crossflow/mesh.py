from argparse import HelpFormatter, Namespace
from dataclasses import dataclass
from math import ceil, pi

import argparse
import math
from pathlib import Path
import gmsh


@dataclass(frozen=True)
class FluidDomain:

    x_min: float
    x_max: float
    y_max: float

    @property
    def y_min(self) -> float:
        return -self.y_max

@dataclass(frozen=True)
class PrismLayer:

    first_layer_height: float
    growth_rate: float
    num_layers: int

    @property
    def total_thickness(self) -> float:
        return self.first_layer_height * (self.growth_rate ** self.num_layers - 1) / (self.growth_rate - 1)

def parse_cli_args(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(
        description='Generate an unstructured quadrangle mesh around a cylinder.',
        formatter_class=lambda prog: HelpFormatter(prog, width=120)
    )
    
    parser.add_argument(
        '--radius',
        dest='radius',
        type=float,
        help='the radius of the cylinder in meters (default: %(default)s m)',
        default=1
    )
    
    parser.add_argument(
        '--x-max',
        dest='x_max',
        type=float,
        help='the maximum X coordinate of the farfield outlet in meters (default: %(default)s m)',
        default=30
    )
    
    parser.add_argument(
        '--x-min',
        dest='x_min',
        type=float,
        help='the minimum X coordinate of the farfield inlet in meters (default: %(default)s m)',
        default=-10
    )
    
    parser.add_argument(
        '--y',
        dest='y',
        type=float,
        help='the positive Y coordinate of the farfield domain in meters (default: %(default)s m)',
        default=10
    )
    
    parser.add_argument(
        '--local-size',
        dest='local_size',
        type=float,
        help='local mesh refinement size near the cylinder in meters (default: %(default)s m)',
        default=0.1
    )
    
    parser.add_argument(
        '--base-size',
        dest='base_size',
        type=float,
        help='base mesh size far from the cylinder in meters (default: %(default)s m)',
        default=1.5
    )
    
    parser.add_argument(
        '--prisms-first-cell-height',
        dest='prisms_first_cell_height',
        type=float,
        help='the thickness of the first prism layer in meters (default: %(default)s m)',
        default=1e-3
    )
    
    parser.add_argument(
        '--prisms-growth',
        dest='prisms_growth',
        type=float,
        help='the geometric growth rate between prism layers (default: %(default)s)',
        default=1.2
    )
    
    parser.add_argument(
        '--num-prisms',
        dest='num_prisms',
        type=int,
        help='number of prism layers near the cylinder (default: %(default)s)',
        default=30
    )
    
    parser.add_argument(
        '--filename',
        dest='filename',
        type=str,
        help='the file path to save the mesh to, must have a \'.msh\' extension',
        default=None
    )

    parser.add_argument(
        '--show-gui',
        dest='show_gui',
        action='store_true',
        help='display the generated mesh'
    )

    return parser.parse_args(argv)

def _check_filename(filename: str | None) -> Path | None:
    if not filename:
        return None
    
    path: Path = Path(filename).expanduser().resolve()
    base_name, _, extension = path.name.rpartition('.')
    return path if base_name and extension == 'msh' else None

def _draw_cylinder_arcs(local_size: float, radius: float) -> tuple[list[int], list[int]]:
    """Generate the points and arcs that form a circle with the specified radius and cell size.

    Four points are used to create a circle with the given radius centered at the origin. The cell sizes near the
    circle will have a local refinement of the specified size.

    :param local_size: The target cell size close to the circle.
    :param radius: The radius of the circle.
    :return: The list of tags defining the points and circular arcs that connect adjacent points together.
    """
    pts: list[int] = []
    for i in range(4):
        theta = 0.5 * pi * i
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        pts.append(gmsh.model.geo.add_point(x, y, 0, local_size))

    center: int = gmsh.model.geo.add_point(0, 0, 0, local_size)
    arcs: list[int] = []
    for i, start in enumerate(pts):
        end = pts[(i + 1) % len(pts)]
        arcs.append(gmsh.model.geo.add_circle_arc(start, center, end))

    return pts, arcs

def _draw_farfield_lines(farfield_size: float, domain: FluidDomain) -> tuple[list[int], list[int]]:
    vertices: list[tuple[float, float]] = [
        (domain.x_max, domain.y_max),
        (domain.x_min, domain.y_max),
        (domain.x_min, domain.y_min),
        (domain.x_max, domain.y_min)
    ]

    pts: list[int] = []
    for x, y in vertices:
        pts.append(gmsh.model.geo.add_point(x, y, 0, farfield_size))

    lines: list[int] = []
    for i, start in enumerate(pts):
        end = pts[(i + 1) % len(pts)]
        lines.append(gmsh.model.geo.add_line(start, end))

    return pts, lines

def _draw_radial_lines(inner_curve: list[int], outer_curve: list[int]) -> list[int]:
    if len(inner_curve) != len(outer_curve):
        raise ValueError(
            f'Different number of points between the inner and outer curves: {len(inner_curve)} inner points and '
            f'{len(outer_curve)} outer points'
        )

    lines: list[int] = []
    for start, end in zip(inner_curve, outer_curve):
        lines.append(gmsh.model.geo.add_line(start, end))

    return lines

def _form_interface_surface(cylinder_arcs: list[int], interface_arcs: list[int], radial_lines: list[int]) -> list[int]:
    surfaces: list[int] = []
    for i, (cylinder_arc, interface_arc, radial_line) in enumerate(zip(cylinder_arcs, interface_arcs, radial_lines)):
        next_line = radial_lines[(i + 1) % len(radial_lines)]
        loop = gmsh.model.geo.add_curve_loop([radial_line, interface_arc, -next_line, -cylinder_arc])
        surface = gmsh.model.geo.add_plane_surface([loop])
        
        gmsh.model.geo.mesh.set_transfinite_surface(surface)
        gmsh.model.geo.mesh.set_recombine(2, surface)
        surfaces.append(surface)

    return surfaces

def _form_farfield_surface(interface_arcs: list[int], farfield_lines: list[int]) -> int:
    interface_loop = gmsh.model.geo.add_curve_loop(interface_arcs)
    farfield_loop = gmsh.model.geo.add_curve_loop(farfield_lines)
    surface = gmsh.model.geo.add_plane_surface([farfield_loop, interface_loop])
    
    gmsh.model.geo.mesh.set_recombine(2, surface)
    return surface

def _ogrid_interface_radius(radius: float, prisms: PrismLayer) -> float:
    """Compute the radius of the interface between the cylinder prism layers and farfield cells.

    :param radius: The cylinder's radius in meters.
    :param prisms: The prism layer settings for the cylinder.
    :return: The radius of the circular O-grid interface in meters.
    """
    return prisms.total_thickness + radius

def _subdivide_interface_region(
    radius: float,
    local_size: float,
    cylinder_arcs: list[int],
    interface_arcs: list[int],
    radial_lines: list[int],
    cylinder_prisms: PrismLayer
):
    num_pts: int = ceil(0.5 * pi * radius / local_size)
    for cylinder_arc, interface_arc, radial_line in zip(cylinder_arcs, interface_arcs, radial_lines):
        gmsh.model.geo.mesh.set_transfinite_curve(cylinder_arc, num_pts + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(interface_arc, num_pts + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(
            radial_line,
            cylinder_prisms.num_layers + 1,
            'Progression',
            cylinder_prisms.growth_rate
        )

def _ignore_constructions(*args: tuple[int, list[int]]):
    construction: list[tuple[int, int]] = []
    for dim, tags in args:
        for tag in tags:
            construction.append((dim, tag))
    
    gmsh.model.set_visibility(construction, False)

def mesh_domain(
    radius: float = 1,
    local_size: float = 0.1,
    base_size: float = 1.5,
    cylinder_prisms: PrismLayer = PrismLayer(0.001, 1.2, 30),
    domain: FluidDomain = FluidDomain(-10, 30, 10),
    filename: str | None = None,
    display: bool = False
):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add('2d_cylinder_compressible_crossflow')

    interface_radius: float = _ogrid_interface_radius(radius, cylinder_prisms)
    cylinder_pts, cylinder_arcs = _draw_cylinder_arcs(local_size, radius)
    interface_pts, interface_arcs = _draw_cylinder_arcs(base_size, interface_radius)
    farfield_pts, farfield_lines = _draw_farfield_lines(base_size, domain)
    radial_lines: list[int] = _draw_radial_lines(cylinder_pts, interface_pts)

    # Divide the cylinder arcs, interface arcs, and radial lines to form the interface region mesh. This method creates
    # a uniform structured mesh around the cylinder that serve as prism layers.
    _subdivide_interface_region(radius, local_size, cylinder_arcs, interface_arcs, radial_lines, cylinder_prisms)

    interface_surfaces: list[int] = _form_interface_surface(cylinder_arcs, interface_arcs, radial_lines)
    farfield_surface: int = _form_farfield_surface(interface_arcs, farfield_lines)
    surfaces: list[int] = interface_surfaces + [farfield_surface]
    gmsh.model.geo.synchronize()
    
    gmsh.model.add_physical_group(2, interface_surfaces + [farfield_surface], name='Fluid')
    gmsh.model.add_physical_group(1, cylinder_arcs, name='Cylinder')
    gmsh.model.add_physical_group(1, farfield_lines, name='Farfield')
    
    # Color mesh elements according to their physical groups
    gmsh.option.set_number('Mesh.ColorCarousel', 2)
    gmsh.option.set_number('Mesh.Algorithm', 8)
    
    # Generate the 2D mesh    
    gmsh.model.mesh.generate(2)
    
    _ignore_constructions((1, interface_arcs), (1, radial_lines), (0, interface_pts))
    if (filename_path := _check_filename(filename)) is not None:
        filename_path.unlink(missing_ok=True)
        filename_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(filename_path))
    
    if display:
        gmsh.fltk.run()
    
    gmsh.finalize()

def main(argv: list[str] | None = None):
    args: Namespace = parse_cli_args(argv)
    domain: FluidDomain = FluidDomain(args.x_min, args.x_max, args.y)
    prisms: PrismLayer = PrismLayer(args.prisms_first_cell_height, args.prisms_growth, args.num_prisms)
    mesh_domain(args.radius, args.local_size, args.base_size, prisms, domain, args.filename, args.show_gui)

if __name__ == '__main__':
    main()