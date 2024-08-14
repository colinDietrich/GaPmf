# This script defines the Sellmeier coefficients and second-order nonlinear optical susceptibility coefficients 
# for different crystal materials, specifically Lithium Niobate (LiNbO3) and Potassium Titanyl Phosphate (KTP).

# Reference:
# - Sellmeier coefficients: https://www.unitedcrystals.com/NLOCOverview.html
# - Second-order susceptibility coefficients: https://www.sciencedirect.com/topics/chemistry/second-order-nonlinear-optical-susceptibility

# Note on notation:
# The second and third indices (j, k) of the d_ijk tensor in second-order nonlinear optics are replaced by a single symbol (l)
# according to the piezoelectric contraction:
# jk: 11  22  33  23,32  31,13  12,21
# l:  1   2   3   4      5      6

# Lithium Niobate (LiNbO3)
LiNbO3 = {
    "name": 'LiNbO3',  # Material name
    "x": [4.9048, 0.11768, 0.04750, 0.027169],  # Sellmeier coefficients for the x-axis (wavelength in micrometers)
    "y": [4.5820, 0.099169, 0.04443, 0.021950],  # Sellmeier coefficients for the y-axis (wavelength in micrometers)
    "d31": 7.11e-12,  # Second-order susceptibility d_31 (type 1 interaction)
    "d22": 3.07e-12,  # Second-order susceptibility d_22 (type 0 interaction)
    "d33": 29.1e-12,  # Second-order susceptibility d_33 (type 0 interaction)
}

# Potassium Titanyl Phosphate Single Crystal (KTP)
KTP = {
    "name": 'KTP',  # Material name
    "x": [3.0065, 0.03901, 0.04251, 0.01327],  # Sellmeier coefficients for the x-axis (wavelength in micrometers)
    "y": [3.0333, 0.04154, 0.04547, 0.01408],  # Sellmeier coefficients for the y-axis (wavelength in micrometers)
    "z": [3.3134, 0.05694, 0.05658, 0.01682],  # Sellmeier coefficients for the z-axis (wavelength in micrometers)
    "d24": 3.64e-12,  # Second-order susceptibility d_24 (type 2 interaction)
    "d31": 2.54e-12,  # Second-order susceptibility d_31 (type 1 interaction)
    "d32": 4.35e-12,  # Second-order susceptibility d_32 (type 1 interaction)
    "d33": 16.9e-12,  # Second-order susceptibility d_33 (type 0 interaction)
}