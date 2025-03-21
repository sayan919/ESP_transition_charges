# ESP Transition Charges

We are looking at the Merz-Kollman (MK) method which involves fitting atomic charges to  
reproduce the Electrostatic Potential (ESP) generated by a Transition Density. By definition, the  
sum of transition charges should be zero, irrespective of the charge of the molecule:

$$
\sum_{i} q^{\text{Tr}}_i = 0
$$

Given grid points $\{R_\mu\}$ with corresponding potentials $\Phi^{QM}_{\mu}$ and atomic positions $\{R_i\}$,  
we want to solve for the atomic charges $\{q_i\}$ such that the electrostatic potential is reproduced and  
the sum of the charges is zero.

Please refer to the derivation_ESP_Transition_Charges.pdf for the full derivation


## Procedure

1. Perform a Vertical Excitation Energy calculation in Gaussian using the following keywords:
   ````
    # CAM-B3LYP/6-31G*  NoSymm TDA(Root=1,NStates=6)  ! Level of theory
    # Integral(Grid=Fine) SCF(Conver=10)              ! Integral and SCF settings
    # scrf=(iefpcm,solvent={env})                     ! for implicit solvent if present
    # IOp(9/40=4)                                     ! Threshold for printing eigenvector components 
    # IOp(6/22=-4)                                    ! select transition density matrix  
    # IOp(6/29=1)                                     ! S1 excited state with IOp(6/22)  
    # IOp(6/30=0)                                     ! S0 ground state with IOp(6/22)  
    # IOp(6/17=2)                                     ! Compute only the electronic contribution
    # Pop=(MK)                                        ! Merz-Kollman method
    # IOp(6/42=12)                                    ! Density of points per unit area in esp fit  
    # IOp(6/33=2)                                     ! Print esp and their positions  
    # IOp(6/14=3)                                     ! Evaluate only the potential at each center
   ````
   Make sure you use these keywords to get the right potential
2. Run the script as follows:
   ````
   python coupling_ATC_ESP.py --log_files monomer1.log monomer2.log
   ````

NOTE: The ESP charges have to be scaled by $\sqrt{2}$ and so couplings by 2. This is implemented
in the code, no extra steps required.
