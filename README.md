matrix manifolds
symmetric positive definite matrix manifolds

===========LONG TERM GOAL=====================================================
Find a method to determine the short term dynamics in protein chains, i.e. a formula

===========SHORT TERM GOAL====================================================
Find a way to mutate some atoms in the chain so that the resulting correlation matrices become vastly different from the Wild Type -> These atoms are more influential

===========PROCEDURE==========================================================

+step 1: simulate 10,000,000 steps to get equilibrium

+step 2: get 1000 configurations near the equilibrium

+step 3: apply centroid method (find point with shortest sum of squared distances to other points) to get a centroid configuration - reference configuration (RC). Note: Align them in Pymol before computing their distance.


+step 4: from RC, run 10,000 simulations, each 1000 mcs steps

+step 5: use mean and pairwise prod means to produce correlation matrices

+step 6: reindex the features (atom coordinates) so that atoms are sorted ascendingly according to their charges

+step 7: use Principal Component Analysis (PCA) to plot the eigenvalues and eigenvalues difference

+step 8 (optional): research matrix manifolds and distances in manifolds to find out other ways to compare correlation matrices

===========TASKS==============================================================

+ Pymol: Know how to align protein chains and visualize their 3D position
+ Centroid method: Know how to extract the 1000 configuration and get the centroid
+ PCA: Know how to plot the eigenvalues (scipy.linalg.eig), how to take the First Principal Vector and their differences
+ Matrix Manifolds: Know how to compute distance in a matrix manifold and tell if two matrix are similar or very different

+ Research 