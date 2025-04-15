# Trajectory Postprocessing and Stability Check Script

The postprocess_traj.py script provides tools for analyzing trajectories by calculating radial distribution functions (RDF) and checking for instability. It uses the RDF mean absolute error (MAE) as a metric to detect instability and processes trajectories accordingly generating a final RDF only using the stable portion and reporting the final MAE respect to the equilibrium traj.

## Features
- **Radial Distribution Function (RDF) Calculation**: Computes the RDF for given trajectory data using binning and a specified maximum radius.
- **Stability Check**: Detects simulation instability by comparing RDFs of successive trajectory segments against a reference.
- **Trajectory Postprocessing**: Identifies and saves the stable portion of the trajectory before instability (if any).
- **MAE Computation**: Computes the RDF mean absolute error (MAE) between trajectory and reference RDFs for detailed analysis. (Averaged over frames)

---

# Usage

Run the script with the following command-line arguments:
```
python script.py --traj_path <TRAJECTORY_FILE> --eq_traj_path <REFERENCE_TRAJECTORY_FILE> [OPTIONS]
```
### Required Arguments:
```
--traj_path: Path to the trajectory file to analyze (e.g., simulation.traj).
--eq_traj_path: Path to the equilibrium (reference) trajectory file (e.g., equilibrium.traj).
```
### Optional Arguments:
```
--nbins: Number of bins for the RDF calculation. Default is 100.
--stability_threshold: MAE threshold to detect simulation instability. Default is 3.0.
--rdf_check_interval: Interval for checking RDF stability (in frames). Default is 20.
```
### Example Command:
```
python script.py --traj_path simulation.traj --eq_traj_path equilibrium_simulation.traj --nbins 100 --stability_threshold 3.0 --rdf_check_interval 20
```
### Outputs
After running, a "stable.rdf" file will be generated containing the rdf of the stable part of the simulation and the final MAE respect to the eq_traj will be printed in the output. 