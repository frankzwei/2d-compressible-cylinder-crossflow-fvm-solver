import argparse

from argparse import HelpFormatter, Namespace
from cylinder_crossflow.mesh import MeshParameters, mesh_domain


def parse_cli_args(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(
        description='numerically solve the flow field for a cylinder in cross-flow.',
        formatter_class=lambda prog: HelpFormatter(prog, width=120)
    )

    return parser.parse_args(argv)

def generate_default_mesh(args: Namespace):
    """Generate a mesh of a cylinder in crossflow using the default mesh parameters."""
    params: MeshParameters = MeshParameters()
    mesh_domain(params, 'build/mesh.msh', False)

def run_simulation(argv: list[str] | None = None):
    args: Namespace = parse_cli_args(argv)

    # Generate an unstructured quadrangle mesh. This mesh is block-structured, meaning the domain is divided into
    # individual quadrants, each are approximately structured, but the global mesh is still unstructured
    generate_default_mesh(args)

if __name__ == '__main__':
    # Simulation inputs and boundary conditions
    gamma: float = 1.4
    r: float = 287
    p_inf: float = 101325
    t_inf: float = 300

    run_simulation()