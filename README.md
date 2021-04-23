# kobumuj
Computes and analyzes cokernels related to Adams operations on certain space and spectra.

Written by Eduardo Fischer
A modified version of this file was included in the author's PhD
dissertation, which may be cited as:
Eduardo Fischer. On endomorphisms of complex cobordism, Adams operations in K-theory, 
and the Im(J) spectrum.  PhD  thesis,  Indiana University, 2021.

Runs in SageMath version 8.8 (Python version 2.7.15)
as well as in SageMath version 9.1 (Python version 3.7.3)

This program is designed to compute and analyze:
i) Adams operations \psi^r on K-homology of CP^infty, BU, BU(n), MU, MU(n)
ii) Cokernel of (\psi^r)-1 on K-homology of CP^infty, BU, BU(n), MU, MU(n)

More information:
p is a fixed odd prime
r is a primitive root of p^2, but often any r coprime with p works
K denotes the spectrum KU_p^hat, p-adic complex K-theory
K_* denotes its homology, K^* its cohomology
d stands for the homological dimension/homological degree
max_j means that we are restricting the problem to the first max_j generatos of K_*(BU)
min_ordinal, max_ordinal means we are restricting the problem to a
few monomials in degrevlex order
k is a parameter to mean the approximation of Z_p^hat by Z/(p^k)Z
For more detailed information, confer the authors' PhD dissertation

Explaining the name:
Kobu means K_0(BU) or K_0(BU(n))
Muj means MU_*J, the complex bordism of the spectrum J, or Im J
Some insight into MU_*J comes through the study of cokernel of Adams operations
For more detailed information, confer the authors' PhD dissertation
