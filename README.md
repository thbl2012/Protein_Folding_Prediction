# Protein Folding Short Term Dynamics
Protein Chain folding simulation. Equilibrium reached when energy is minimum. Chain is updated with Metropolis method.

## Long Term Goal
Find a method to determine the short term dynamics in protein chains, i.e. a formula.

## Short Term Goal
Find a way to mutate some atoms in the chain so that the resulting correlation matrices become vastly different from the Wild Type &rarr; These atoms are more influential.

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

- [x] Pymol: Know how to align protein chains and visualize their 3D position.
- [x] Centroid method: Know how to extract the 1000 configuration and get the centroid.
- [x] PCA: Know how to plot the eigenvalues (scipy.linalg.eig), how to take the First Principal Vector and their differences.
- [x] Matrix Manifolds: Know how to compute distance in a matrix manifold and tell if two matrix are similar or very different.
- [x] Keywords: matrix manifolds, symmetric positive definite matrix manifolds.

## Task List 1

- [ ] Do 100-10000 wild type simulations with different random seeds.
- [ ] Change charges to the following:
2 2 2 2 2...
0 0 0 0 0 ...
2 -2 2 -2 2 -2 ...
- [ ] Find some way to interpolate to 0 2 0 -2 ...
- [ ] Do also do 1000 simulations for each mutant
- [ ] Get data
- [ ] Create a neural network to classify each type of mutant based on their matrices

## Task List 2

### For wild_type:
1. Free all atoms
2. Repeat 5 times save all images put side by side
3. try:
3.1. Run only 101 mcs
3.2. Repeat only for 10,100 times for the correlation
3.3. Generate 500 correlation matrices

### Charge sequences:
1. 2-0--2-0-2... WT
2. 2-2-2-2...
3. 2-0-2-0...
4. 0-0-0-0...
5. 0*n - WT
6. 2*n - WT
7. -2*n - WT
8. WT -0*n - WT
9. WT - 2*n - WT

### Research about:
- [ ] Convolutional neural network

## Problems
### Simulation
- [ ] Possibly always stuck in a local minimum
- [ ] Try Bayesian Optimization (Break into 5-6 subchains) to find minimum
### Correlation
- [ ] Most features (coordinates) never changed fast enough to collect meaningful correlations
- [ ] Try:
  - Run 10,000 - 1,000,000 short simulations instead of only 1000
  - Transform correlation formula to bypass the number of simulations --> avoid errors with too small numbers
