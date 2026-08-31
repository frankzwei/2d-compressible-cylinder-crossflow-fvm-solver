# 2D Cylinder in Crossflow Finite Volume Solver

This program simulates crossflow of a compressible gas across a cylinder in two dimensions. The solver internally solves the compressible Navier-Stokes equations using the finite volume approach to resolve the flow field. For ease of operation, the solver uses the Steger-Warming flux splitting scheme to evaluate the fluxes entering and exiting each cell.

## Meshing

The mesh is generated automatically by the program. Users may generate the mesh manually to view and inspect it by running `uv run mesh`. The CLI arguments can be viewed using `uv run mesh --help` and permit the user to apply custom parameters to adjust the resulting mesh.

The mesher generates a block-structured quad mesh. This mesh is globally unstructured as certain cells, particularly the transition region from a circular pattern to rectangular pattern, border more than four cells. However, the mesh is called a "block-structured" mesh as the flow field is divided into nine blocks. Each block contains a structured mesh. Laplacian smoothing is applied to the raw mesh to improve element quality and reduce skewness of the cells in the transition from cylindrical to rectangular.

## Solving

As mentioned above, the finite volume solver uses Steger-Warming flux splitting scheme to compute the fluxes into and out of each cell.

# Development

The Python program uses `uv` to create and manage its virtual environment. The native Python version used for development is `Python 3.11`. To get set up for development, follow the steps below.

1. Download `uv`, either from a pre-existing Python version or the standalone installer from the `uv` website.

2. Clone the Github repository using `git clone git@github.com:frankzwei/2d-compressible-cylinder-crossflow-fvm-solver.git`.

3. Open the project root directory in an IDE (e.g., VS Code) and run `uv sync`. This will automatically create the virtual environment and download the relevant package dependencies.

To test that the installation was successful, display the `--help` menu for the solver and mesher.