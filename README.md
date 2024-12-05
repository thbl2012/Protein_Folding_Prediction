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
5. Use mean and pairwise product means to produce correlation matrices.
6. Reduce the dimension of correlation matrices by taking the three most significant diagonals.
7. Build a Convolutional Neural Network to predict if the original chain is a mutant type using the above input data.
8. Report results.

## Report

For the detailed report, see [Protein_Folding_Report.pdf](Protein_Folding_Report.pdf)
