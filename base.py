import numpy as np
from math import tau, sin, cos
import openmeteo_requests as om_requests  # see API docs at https://open-meteo.com/en/docs/climate-api
import pandas as pd
from matplotlib import pyplot as plt
from typing import Any


AXIAL_TILT: float = 23.5 * tau / 360
SOLAR_CONSTANT: float = 1373.0
EVAL_PTS: int = 2 ** 16
CLIMATE_MODEL_METRICS: list[str] = ["temperature_2m_mean", "cloud_cover_mean", "relative_humidity_2m_mean"]
CLIMATE_MODEL_METRIC_UNITS: list[str] = ["°C", "%", "%"]
LOCATION_DICT: dict[str: tuple[float, float]] = {
		"New York, NY": (40.714, -74.006),
		"Chapel Hill, NC": (35.913, -79.056),
		"Juneau, AK": (58.305, -134.433),
		"Patagonia, Argentina": (-41.810, -68.906),
		"Helsinki, Finland": (60.192, 24.946),
		"South Pole": (-90.000, 0.000),
		"North Pole": (90.000, 0.000),}


def direct_solar_radiation(latitude_rad: float, longitude_rad: float, n_t: int = EVAL_PTS) -> np.array:
	"""
	Computes direct solar radiation over the course of one year for the location defined by (latitude_rad, longitude_rad)
	:param latitude_rad: Latitude of location of interest, in radians
	:param longitude_rad: Longitude of location of interest, in radians
	:param n_t: Number of equally spaced timepoints to compute direct solar radiation for
	:return: Direct solar radiation intensity
	"""

	year_fraction_vector: np.array = np.linspace(0, 1, n_t)
	day_fraction: np.array = tau * ((year_fraction_vector * 365.24 + longitude_rad / tau) % 1)
	solar_axial_offset: np.array = (AXIAL_TILT * np.cos(tau * year_fraction_vector - latitude_rad))

	sun_vecx = np.sin(day_fraction) * np.cos(solar_axial_offset)
	sun_vecy = np.cos(day_fraction) * np.cos(solar_axial_offset)
	sun_vecz = np.sin(solar_axial_offset)

	earth_vecx: float = sin(longitude_rad) * cos(-latitude_rad)
	earth_vecy: float = cos(longitude_rad) * cos(-latitude_rad)
	earth_vecz: float = sin(-latitude_rad)

	cos_angle_vec: np.array = sun_vecx * earth_vecx + sun_vecy * earth_vecy + sun_vecz * earth_vecz
	incident_angle_vec: np.array = np.acos(cos_angle_vec)

	intensity_vec_no_night: np.array = SOLAR_CONSTANT * cos_angle_vec
	# mask for only daylight hours, where the sun is above the horizon; set other values to 0
	intensity_vec_out: np.array = intensity_vec_no_night * (incident_angle_vec < tau / 4)

	return intensity_vec_out


def get_climate_data(latitude_deg: float, longitude_deg: float) -> pd.DataFrame:
	"""
	Gets synthetic climate data from Open Meteo using the EC_Earth3P_HR model
	:param latitude_deg: Latitude of the location of interest, in degrees
	:param longitude_deg: Longitude of the location of interest, in degrees
	:return: DataFrame of synthetic climate data
	"""

	url = "https://climate-api.open-meteo.com/v1/climate"
	params: dict[str: Any] = {
		"latitude": latitude_deg,
		"longitude": longitude_deg,
		"start_date": "2025-01-01",
		"end_date": "2049-12-31",
		"models": ["EC_Earth3P_HR",],  # this model has the best metrics for our purposes imo
		"daily": CLIMATE_MODEL_METRICS,
	}

	api_client = om_requests.Client()
	responses = api_client.weather_api(url, params=params)
	response = responses[0]  # we have to specify the 0th index because the api is meant for batching over many locations

	daily_data = response.Daily()
	dt_index: pd.DatetimeIndex = pd.date_range(
		start=pd.to_datetime(daily_data.Time(), unit="s", utc=True),
		end=pd.to_datetime(daily_data.TimeEnd(), unit="s", utc=True),
		freq=pd.Timedelta(seconds=daily_data.Interval()),
		inclusive="left",
	)
	df_out = pd.DataFrame()
	df_out.index = dt_index

	for i, metric in enumerate(CLIMATE_MODEL_METRICS):
		metric_values = daily_data.Variables(i).ValuesAsNumpy()
		df_out[metric] = metric_values

	return df_out


def compute_climate_data_annual_means(data_df: pd.DataFrame):
	"""
	Computes column-wise means of data_df grouped by day of year
	:param data_df:
	:return: vector of mean values of each column of data_df for each day of the year
	"""
	data_df["day of year"] = data_df.index.day_of_year
	daily_means = data_df.groupby("day of year").mean()
	return daily_means


def get_mean_intensities(timeseries: np.array) -> float:
	"""
	Does quadrature to compute the mean intensity of a solar intensity timeseries
		Timeseries step size must be equal
	:param timeseries: Data to do quadrature on
	:return:
	"""
	n_pts: int = np.size(timeseries)
	quad_chunks: np.array = (timeseries[1:] + timeseries[:-1]) * 0.5
	quad: float = np.sum(quad_chunks) / (n_pts - 1)

	return quad


def toolchain(latitude_deg: float, longitude_deg: float, placename: str | None = None):
	"""
	Combines the solar radiation analysis toolset to compute direct, diffuse and reflected radiation at the location
		(latitude_deg°, longitude_deg°)
	:param latitude_deg:
	:param longitude_deg:
	:param placename: Optional parameter for plots; the name of the location specified by (latitude_deg, longitude_deg)
	:return:
	"""

	latitude_rad: float = latitude_deg * tau / 360
	longitude_rad: float = longitude_deg * tau / 360

	g_dir = direct_solar_radiation(latitude_rad, longitude_rad)
	g_dir_mean: float = get_mean_intensities(g_dir)
	g_dir_daylight_mean: float = get_mean_intensities(g_dir[g_dir > 0])

	historical_climate_data = get_climate_data(latitude_deg, longitude_deg)
	data_daily_means = compute_climate_data_annual_means(historical_climate_data)

	annualized_output_kwh = g_dir_mean * 60 * 60 * 24 * 365.24 / 3_600_000
	print(
		f"Predicted annual output: {round(annualized_output_kwh, 3)} kWh"
	)
	make_plots(g_dir, g_dir_mean, g_dir_daylight_mean, data_daily_means, placename)

	return


def make_plots(
		radiation_intensity: np.array,
		mean_radiation_intensity: float,
		mean_daylight_radiation_intensity: float,
		simulation_mean_df: pd.DataFrame,
		title: str | None = None,
):
	"""
	Plots relevant parameters
	:param radiation_intensity:
	:param mean_radiation_intensity:
	:param mean_daylight_radiation_intensity:
	:param simulation_mean_df:
	:param title:
	:return:
	"""

	fig, [[ax00, ax01], [ax10, ax11]] = plt.subplots(2, 2)

	ax00.plot(np.linspace(0, 1, EVAL_PTS) * 365, radiation_intensity, label="Direct solar intensity")
	ax00.axhline(y=mean_radiation_intensity, xmin=0, xmax=1, linestyle="--", label="Mean intensity", color=(1, 0, 1))
	ax00.axhline(y=mean_daylight_radiation_intensity, xmin=0, xmax=1, linestyle="--", label="Mean daylight intensity", color=(1, 0, 0.3))
	ax00.set_xlabel("Day of year")
	ax00.set_ylabel("Intensity (W m^-2)")
	ax00.legend()

	metric = CLIMATE_MODEL_METRICS[0]
	units = CLIMATE_MODEL_METRIC_UNITS[0]
	ax01.plot(np.arange(start=1, stop=367), simulation_mean_df[metric], label=metric)
	ax01.set_xlabel("Day of year")
	ax01.set_ylabel(units)
	ax01.legend()

	metric = CLIMATE_MODEL_METRICS[1]
	units = CLIMATE_MODEL_METRIC_UNITS[1]
	ax10.plot(np.arange(start=1, stop=367), simulation_mean_df[metric], label=metric)
	ax10.set_xlabel("Day of year")
	ax10.set_ylabel(units)
	ax10.legend()

	metric = CLIMATE_MODEL_METRICS[2]
	units = CLIMATE_MODEL_METRIC_UNITS[2]
	ax11.plot(np.arange(start=1, stop=367), simulation_mean_df[metric], label=metric)
	ax11.set_xlabel("Day of year")
	ax11.set_ylabel(units)
	ax11.legend()

	if title:
		fig.suptitle(title)

	plt.show()

	return


if __name__ == '__main__':

	place = "Chapel Hill, NC"
	toolchain(*LOCATION_DICT.get(place), placename=place)

	from formulas import bb_spectral_irradiance, integrate_quad
	from math import pi
	from constants import P_SE, R_SUN, sb, T_SUN, R_E, R_E_SUN

	# example 2: Blackbody spectral irradiance
	# for wavelengths 10 nm to 3µm
	"""start, end = 1e-8, 3e-6
	# vectorize in 1nm increments
	vec_in = np.linspace(start, end, int((end - start) // 1e-9))
	# compute irradiance at each wavelength
	irradiance = bb_spectral_irradiance(vec_in)
	# integrate power over the whole spectrum
	net_power_m2 = integrate_quad(vec_in, irradiance)
	print(f"{net_power_m2 = }")
	# multiply by sun surface area to get solar power;
	net_power = net_power_m2 * 4 * pi * R_SUN**2
	print(f"{net_power = }")
	# divide by earth's arc fraction to get solar power recieved by earth's atmosphere, on average
	net_power_recieved = net_power * (2 * pi * R_E**2) / (pi * R_E_SUN**2)
	print(f"{net_power_recieved = }")
	# plot
	plt.plot(vec_in, irradiance)
	plt.show()"""

	# print total power
	# print(4 * pi * R_SUN ** 2 * sb * T_SUN ** 4 * P_SE)

