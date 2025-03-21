"""
Author : Sayan Adhikari and Christopher Mayers
Email  : sayanadh919@gmail.com
Phone  : +1 (209)-455-6739
Place  : University of California, Merced
Lab    : Dr. Isborn
Code   : Calculate the excitonic coupling between two monomers using the transition charges
        obtained from the electrostatic potential fitting
Usage  : python coupling_tQ_ESP.py --log_files monomer1.log monomer2.log
"""


import sys
import numpy as np
import argparse
import re

ANGS_TO_BOHRS = 1.8897259885789
HA_TO_EV = 27.211396132

def print_banner(title, char="=", width=80):
    """
    Print a banner with the specified title.

    Args:
        title (str): The title of the banner.
        char (str, optional): The character used to create the border of the banner. Default is '='.
        width (int, optional): The width of the banner. Default is 80.
    """
    print(char * width)
    print(title.center(width))
    print(char * width)


# Print the banner
print_banner("Coulombic Coupling using Transition Charges (ESP Fitting)")

def extract_symbols_and_coordinates(file_path):
    symbols = []
    coordinates = []
    with open(file_path, 'r') as file:
        read_data = False
        for line in file:
            # Start reading after the 'Symbolic Z-matrix:' keyword
            if 'Symbolic Z-matrix:' in line:
                read_data = True
                continue
            if read_data:
                # Stop at an empty line or another header
                if line.strip() == '':
                    break
                # Match lines with atomic symbols followed by coordinates
                match = re.match(r'^\s*([A-Z][a-z]?)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)', line)
                if match:
                    symbol, x, y, z = match.groups()
                    symbols.append(symbol)
                    coordinates.append((float(x), float(y), float(z)))
    return symbols, np.array(coordinates)ß

def extract_grid(input_file):
    grid = []
    with open(input_file, 'r') as file:
        for line in file:
            if "ESP Fit Center" in line:
                parts = line.split()
                coords = [float(parts[6]), float(parts[7]), float(parts[8])]
                grid.append(coords)
    return np.array(grid)

def extract_potential(input_file):
    potential = []
    start_pattern = "Electrostatic Properties (Atomic Units)"
    start_reading = False

    with open(input_file, 'r') as file:
        for line in file:
            if start_pattern in line:
                start_reading = True
                continue
            if start_reading and "Fit" in line:
                parts = line.split()
                potential_value = float(parts[2])
                potential.append(potential_value)
    return np.array(potential)

def extract_dipole_moments(input_file):
    start_pattern = "Ground to excited state transition electric dipole moments (Au):"
    state_pattern = "1"
    start_reading = False
    dipole_moments = []

    with open(input_file, 'r') as file:
        for line in file:
            if start_pattern in line:
                start_reading = True
                next(file)  # Skip the next line after finding the start pattern
                continue
            if start_reading and state_pattern in line:
                parts = line.split()
                try:
                    dipole_moments = [float(parts[1]), float(parts[2]), float(parts[3])]
                    break
                except ValueError:
                    continue
    return dipole_moments

def esp_charges(coordinates, grid, potential):
    N = len(coordinates)
    e = np.zeros(N)
    for i in range(N):
        e[i] = np.sum(potential / np.linalg.norm(grid - coordinates[i], axis=1))
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            G[i, j] = np.sum(1 / (np.linalg.norm(grid - coordinates[i], axis=1) * 
                                  np.linalg.norm(grid - coordinates[j], axis=1)))           
    A = np.block([[G, np.ones((N, 1))],
                  [np.ones((1, N)), 0]])
    Q = [0.0]
    b = np.concatenate([e, Q])
    solution = np.linalg.solve(A, b)
    q = solution[:N]
    c = np.sum(potential**2)
    msd = c - 2*q.T @ e + q.T @ G @ q
    rrmsd = np.sqrt(msd/c)
    return q, rrmsd


def coupling_via_TC(q_monomer1: np.ndarray, q_monomer2: np.ndarray,
                  coordinates_1: np.ndarray, coordinates_2: np.ndarray) -> float:
    
    J = 0
    for i in range(len(q_monomer1)):
        for j in range(len(q_monomer2)):
            J += (q_monomer1[i] * q_monomer2[j])/(np.linalg.norm(coordinates_2[j] - coordinates_1[i]))

    return J* HA_TO_EV


def main():
    parser = argparse.ArgumentParser(description='Generate Gaussian input files for monomers at different intermonomer distances.')
    parser.add_argument('--log_files', type=str, nargs=2, required=True, help='Path to the XYZ file of the dimer')
    args = parser.parse_args()

    log_file1 = args.log_files[0]
    log_file2 = args.log_files[1]

    atoms_monomer1, positions_monomer1 = extract_symbols_and_coordinates(log_file1)
    atoms_monomer2, positions_monomer2 = extract_symbols_and_coordinates(log_file2)

    positions_monomer1 = positions_monomer1*ANGS_TO_BOHRS
    positions_monomer2 = positions_monomer2*ANGS_TO_BOHRS

    grid_monomer1 = extract_grid(log_file1)*ANGS_TO_BOHRS
    grid_monomer2 = extract_grid(log_file2)*ANGS_TO_BOHRS

    potential_monomer1 = extract_potential(log_file1)
    potential_monomer2 = extract_potential(log_file2)

    gaussian_tdp_monomer1 = extract_dipole_moments(log_file1)
    gaussian_tdp_monomer2 = extract_dipole_moments(log_file2)

    q_monomer1, RRMSD1 = esp_charges(positions_monomer1, grid_monomer1, potential_monomer1)
    q_monomer2, RRMSD2 = esp_charges(positions_monomer2, grid_monomer2, potential_monomer2)

    calculated_tdp_monomer1 = np.sum(q_monomer1[:, None] * positions_monomer1, axis=0)
    calculated_tdp_monomer2 = np.sum(q_monomer2[:, None] * positions_monomer2, axis=0)

    print('MONOMER 1\n')
    for i in range(len(q_monomer1)):
        print(f'{atoms_monomer1[i]}: {q_monomer1[i]:8.4f}')
    print(f'\nTotal charge: {np.sum(q_monomer1):8.4f}')
    print(f'RRMSD: {RRMSD1:8.4f}')    
    print(f'\nCalculated Dipole:      {calculated_tdp_monomer1[0]:8.4f} {calculated_tdp_monomer1[1]:8.4f} {calculated_tdp_monomer1[2]:8.4f}')
    print(f'Calc_dipole/sqrt(2):    {calculated_tdp_monomer1[0]/np.sqrt(2):8.4f} {calculated_tdp_monomer1[1]/np.sqrt(2):8.4f} {calculated_tdp_monomer1[2]/np.sqrt(2):8.4f}')
    print(f'Gaussian Dipole:        {gaussian_tdp_monomer1[0]:8.4f} {gaussian_tdp_monomer1[1]:8.4f} {gaussian_tdp_monomer1[2]:8.4f}')


    print('\nMONOMER 2\n')
    for i in range(len(q_monomer2)):
        print(f'{atoms_monomer2[i]}: {q_monomer2[i]:8.4f}')
    print(f'\nTotal charge: {np.sum(q_monomer2):8.4f}')
    print(f'RRMSD: {RRMSD2:8.4f}')

    print(f'\nCalculated Dipole:      {calculated_tdp_monomer2[0]:8.4f} {calculated_tdp_monomer2[1]:8.4f} {calculated_tdp_monomer2[2]:8.4f}')
    print(f'Calc_dipole/sqrt(2):    {calculated_tdp_monomer2[0]/np.sqrt(2):8.4f} {calculated_tdp_monomer2[1]/np.sqrt(2):8.4f} {calculated_tdp_monomer2[2]/np.sqrt(2):8.4f}')
    print(f'Gaussian Dipole:        {gaussian_tdp_monomer2[0]:8.4f} {gaussian_tdp_monomer2[1]:8.4f} {gaussian_tdp_monomer2[2]:8.4f}')

    J = coupling_via_TC(q_monomer1/np.sqrt(2), q_monomer2/np.sqrt(2), positions_monomer1, positions_monomer2)
    print(f"\nCoulombic Coupling via Transition Charges: {J:.4f} eV")

if __name__ == "__main__":
    main()