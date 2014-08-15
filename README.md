rsw_1L
======
Based off of Francis Poulin's rsw_1L code

sw_1L_channel_np2ps_stab.py  
>Creates A using numpy/scipy then moves rows to petsc4py matrix Could use eig, eigs, or slepc 

sw_1L_channel_ps_fixed.py 
>A (petsc) built with separate parts, using the other numpy matrices (Dy, dH, etc.) 

sw_1L_channel_np_stab2.py and sw_1L_channel_n_stab.py 
>Basically, just use numpy only. 2 ways of making A (2nd is better/faster) 
>>sw_1L_channel_np_stab2 uses numpy only .. can choose between eig and eigs. Converted some matrices to be sparse. Builds A for each piece.
>>sw_1L_channel_np_stab -- numpy only, eig/eigs, builds matrix by concatenating each part.

sw_1L_channel_ps_newguess.py 
>petsc only – guess changes to be the previous one just calculated. Still need an initial guess.vvv

sw_1L_mat_free.py
>Uses matrix free method, but only works in serial.
>Use these options: -eps_target 0.21,0.45 -nev 100 -eps_ncv 120 //nev has to change if you have a high Ny, default Ny=100.
>Lower tolerance to increase speed (more error though).
>EPS will always go to max_it, you can reduce this if you find the amount of eigenvalues you want already.

rsw_1L_ps_fixed_io.py
> Creates files to store parameters, eigenvalues, and eigenvectors. After running this, run read_rsw_1L.py to see the plot
