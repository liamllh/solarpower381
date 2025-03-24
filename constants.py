from math import tau

# physical constants
h: float = 6.626_070_150e-34  # J s
h_bar: float = h / tau  # J s
c: float = 299_792_458  # m s^-1
k: float = 1.380_649e-23  # J K^-1
sb: float = 5.670_374_419e-8  # W m^-2 K^-4

# measured astronomical constants
T_SUN: float = 5773  # K
R_E_SUN: float = 149.6e9  # m
R_SUN: float = 695.7e6  # m
R_E: float = 6.371e6  # m

# derived constants
P_SE: float = R_E**2 / R_E_SUN**2  # fractional power of sun absorbed by earth; 1.813643915e-9
