import numpy as np
from constants import h, h_bar, c, k, T_SUN
from math import pi


MAX_EXPONENT: float = 32.0
MIN_WAVELENGTH_5: float = 1e-30


def bb_spectral_irradiance(wavelength_vec_m: np.array) -> np.array:
	"""
	Blackbody spectral irradiance for an array of wavelengths, in W m^-2 m^-1 (power per area per wavelength)
	*** Note that f32 rounding errors require inputs to be greater than 4e-9 / 4 nm ***
	:param wavelength_vec_m: Vector of wavelengths in m
	:return out: spectral irradiance at each wavelength
	"""

	# we define things in this specific order to try to avoid overflows for nm to micron scale lambda
	c1 = h_bar * c**2  # O(1e-17)
	lambda5 = wavelength_vec_m**5  # O(1e-30 - 1e-45)
	coeff = c1 / lambda5  # O(1e13 - 1e28)

	bolzmann = k * T_SUN  # O(1e-20)
	exp_coeff = h * c / bolzmann  # O(1e-6)
	exp_arg = np.clip(exp_coeff / wavelength_vec_m, a_min=None, a_max=MAX_EXPONENT)  # O(1 - 1e3), clipped to (1 - 64)
	exp_part = np.exp(exp_arg) - 1  # O(1 - 1e28)

	out = coeff / exp_part  # O(1e-16 - 1e27)

	return out


def rayleigh_cross_section_m2(wavelength_vec_m: np.array, gas_refractive_index: float, gas_radius_m: float) -> np.array:
	"""
	Unfinished function
	:param wavelength_vec_m:
	:param gas_refractive_index:
	:param gas_radius_m:
	:return:
	"""
	coeff: float = 16.0 * pi**2.0 / 3.0
	n2: float = gas_refractive_index**2
	refractive_index_factor: float = ((n2 - 1) / (n2 + 2))**2
	arr_out: np.array = gas_radius_m**6 / wavelength_vec_m
	arr_out = arr_out * refractive_index_factor * coeff

	return arr_out


def integrate_quad(xs: np.array, ys: np.array) -> float:
	"""
	Do quadrature to integrate xs d(ys) from xs[0] to xs[-1]
	:param xs: x data
	:param ys: y data
	:return: integral value
	"""
	midpoints: np.array = (ys[1:] + ys[:-1]) * 0.5
	dx: np.array = xs[1:] - xs[:-1]
	i: float = np.asarray(np.sum(midpoints * dx), float)
	return i


if __name__ == "__main__":

	from matplotlib import pyplot as plt
	arr_in = np.linspace(1e-8, 3e-6, 900)
	irradiance = bb_spectral_irradiance(arr_in)
	plt.plot(arr_in, irradiance * arr_in)
	plt.show()
	pass
