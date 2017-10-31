# Protein Folding Short Term Dynamics
Protein Chain folding simulation. Equilibrium reached when energy is minimum.

## Long Term Goal
Find a method to determine the short term dynamics in protein chains, i.e. a formula.

## Short Term Goal
Find a way to mutate some atoms in the chain so that the resulting correlation matrices become vastly different from the Wild Type &rarr These atoms are more influential.

## Procedure

1. Simulate 10,000,000 steps to get equilibrium.
2. Get 1000 configurations near the equilibrium.
3. Apply centroid method (find point with shortest sum of squared distances to other points) to get a centroid configuration reference configuration (RC). Note: Align them in Pymol before computing their distance.
4. From RC, run 10,000 simulations, each 1000 mcs steps.
5. Use mean and pairwise prod means to produce correlation matrices.
6. Reindex the features (atom coordinates) so that atoms are sorted ascendingly according to their charges.
7. Use Principal Component Analysis (PCA) to plot the eigenvalues and eigenvalues difference.
8. (Optional) Research matrix manifolds and distances in manifolds to find out other ways to compare correlation matrices.

## Tasks

[] Pymol: Know how to align protein chains and visualize their 3D position.
[] Centroid method: Know how to extract the 1000 configuration and get the centroid.
[] PCA: Know how to plot the eigenvalues (scipy.linalg.eig), how to take the First Principal Vector and their differences.
[] Matrix Manifolds: Know how to compute distance in a matrix manifold and tell if two matrix are similar or very different.
[] Keywords: matrix manifolds, symmetric positive definite matrix manifolds.
