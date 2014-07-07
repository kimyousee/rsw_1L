rsw_1L
======

sw_1L_channel_np2ps_stab.py  Creates A using numpy/scipy then moves rows to petsc4py matrix Could use eig, eigs, or slepc sw_1L_channel_ps_fixed.py A (petsc) built with separate parts, using the other numpy matrices (Dy, dH, etc.) sw_1L_channel_np_stab2.py and sw_1L_channel_n_stab.py see v5.2 above basically, just use numpy only. 2 ways of making A (2 is better/faster) sw_1L_channel_ps_newguess.py petsc only – guess changes to be the previous one just calculated. Still need an initial guess.vvv
