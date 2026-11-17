"""Worked example for converting field-level net irrigation demand into boundary-specific
gross requirements on a radial open-canal network.

The package implements the standard propagation equations (farm-gate demand, edge inflows,
head requirement), the partition of losses into consumed, recoverable and non-recoverable
parts, water-balance closure at each system boundary, two efficiency-coefficient benchmarks,
Monte Carlo uncertainty propagation and grouped Sobol sensitivity indices.
"""

__version__ = "0.1.0"
