import numpy as np
from math import tau, sin, cos
import openmeteo_requests as om_requests  # see API docs at https://open-meteo.com/en/docs/climate-api
import pandas as pd
from matplotlib import pyplot as plt
from typing import Any
from constants import deg_rad
from scipy.stats import linregress


CLOUD_TRANSMISSIVITY_AT_100: float = 0.6
AXIAL_TILT: float = 23.5 * deg_rad
SOLAR_CONSTANT: float = 1373.0
EVAL_PTS: int = 2 ** 16
T_STC: float = 298.15
w_day_to_kwh_year = 365 * 24 * 3_600 / 3_600_000
CLIMATE_MODEL_METRICS: list[str] = ["temperature_2m_mean", "cloud_cover_mean", "relative_humidity_2m_mean", "pressure_msl_mean", "temperature_2m_min", "temperature_2m_max"]
CLIMATE_MODEL_METRIC_UNITS: list[str] = ["°C", "%", "%", "kPa?", "°C", "°C"]
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
	day_fraction: np.array = tau * ((year_fraction_vector * 365 + longitude_rad / tau) % 1)
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


def diffuse_solar_radiation(
		latitude_deg: float,
		n_t: int = EVAL_PTS,
) -> np.array:

	year_fractions: np.array = np.linspace(0, 1, n_t)
	day_fractions: np.array = (365 * year_fractions) % 1


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


def m_theta(latitude_rad: float) -> np.array:
	# optical air mass at solar zenith angle theta
	year_fraction_vector: np.array = np.linspace(0, 1, EVAL_PTS)
	solar_axial_offset: np.array = (AXIAL_TILT * np.cos(tau * year_fraction_vector - latitude_rad))
	return sin(latitude_rad) * np.sin(solar_axial_offset) + cos(latitude_rad) * np.cos(solar_axial_offset)


def mean_annual_over_days(annual_timeseries: np.array) -> np.array:
	"""
	Returns average value for each day of a year long timeseries
	:param annual_timeseries:
	:return:
	"""

	binned_data = split_days(annual_timeseries)
	out = np.concat(binned_data, axis=1)
	out = np.average(out, axis=0)

	return out


def mean_nonzero_annual_over_days(annual_timeseries: np.array) -> np.array:

	binned_data = split_days(annual_timeseries)
	data_out: list[np.array] = list()
	for arr in binned_data:
		arr[arr == 0] = np.nan
		data_out.append(arr)

	out = np.array([np.mean(arr[~np.isnan(arr)]) for arr in data_out])
	return out


def split_days(annual_timeseries: np.array) -> list[np.array]:

	binsize = annual_timeseries.size // 365
	out_list: list[np.array] = list()
	idx = 0
	for n in range(365):
		out_list.append(annual_timeseries[idx: idx + binsize, None])
		idx += binsize

	return out_list


def average_nonzero_value(arr: np.array):
	arr[arr == 0] = np.nan
	return np.mean(arr)


def match_annualized_to_timeseries(
		annualized_data: np.array,  # size 365
		timeseries: np.array,
) -> np.array:
	out = np.concat([annualized_data[day - 2, None] for day in timeseries.day_of_year])
	return out


def max_daily_transmittance(dir_sol_rad, latitude_rad, vapor_pressure: np.array) -> np.array:

	optical_air_mass = m_theta(latitude_rad)
	# assuming at sea level
	tau_inst_sea_level = 0.87
	dsr_mtheta = dir_sol_rad * (tau_inst_sea_level**optical_air_mass)

	dsr_daily = mean_annual_over_days(dir_sol_rad)
	dsr_mtheta_daily = mean_annual_over_days(dsr_mtheta)

	alpha = -0.061

	out = dsr_mtheta_daily / dsr_daily + alpha * vapor_pressure


def compute_climate_data_doy_means(data_df: pd.DataFrame):
	"""
	Computes column-wise means of data_df grouped by day of year
	:param data_df:
	:return: vector of mean values of each column of data_df for each day of the year
	"""
	data_df["day of year"] = data_df.index.day_of_year
	daily_means = data_df.groupby("day of year").mean()
	return daily_means


def compute_climate_data_annual_means(data_df: pd.DataFrame):
	"""
	Computes column-wise means of data_df grouped by day of year
	:param data_df:
	:return: vector of mean values of each column of data_df for each day of the year
	"""
	data_df["year"] = data_df.index.year
	daily_means = data_df.groupby("year").mean()
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
	quad: float = (np.sum(quad_chunks)[0]) / (n_pts - 1)

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
	data_daily_means = compute_climate_data_doy_means(historical_climate_data)

	annualized_output_kwh = g_dir_mean * 60 * 60 * 24 * 365 / 3_600_000
	print(
		f"Predicted annual output: {round(annualized_output_kwh, 3)} kWh"
	)
	make_daily_mean_plots(g_dir, g_dir_mean, g_dir_daylight_mean, data_daily_means, placename)

	return


def make_daily_mean_plots(
		radiation_intensity: np.array,
		mean_radiation_intensity: float,
		mean_daylight_radiation_intensity: float,
		simulation_mean_df: pd.DataFrame,
		title: str | None = None,
):
	"""
	Plots averages over each day of the year of:
		direct radiation intensity 	|	temp
		could cover					| 	relative humidity
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


def make_annual_mean_plots(
		radiation_intensity: np.array,
		mean_radiation_intensity: float,
		mean_daylight_radiation_intensity: float,
		simulation_mean_df: pd.DataFrame,
		title: str | None = None,
):
	"""
	Plots averages over each day of the year of:
		direct radiation intensity 	|	temp
		could cover					| 	relative humidity
	:param radiation_intensity:
	:param mean_radiation_intensity:
	:param mean_daylight_radiation_intensity:
	:param simulation_mean_df:
	:param title:
	:return:
	"""
	return


def wraparound_ma(arr: np.array, window_size: int) -> np.array:
	slices = [slice(n - window_size, n) for n in range(window_size)]
	to_mean = [arr[this_slice, None] for this_slice in slices]
	out = np.mean(to_mean, axis=0)
	return out


class SolarArray(object):

	def __init__(
			self,
			# place variables
			latitude_deg: float,
			longitude_deg: float,
			# solar cell/array variables
			eta_tref: float = 0.2,  # unitless; STC efficiency
			beta_tref: float = 0.004,  # per Kelvin; temperature coefficient
			t_stc: float = T_STC,  # Kelvin; test temperature conditions
			arr_area_m2: float = 1,
			placename: str | None = None,
	):
		# initialize place
		self.latitude_deg = latitude_deg
		self.latitude_rad = self.latitude_deg * deg_rad
		self.longitude_deg = longitude_deg
		self.longitude_rad = self.longitude_deg * deg_rad
		self.placename = placename
		# initialize panel properties
		self.eta_ref = eta_tref
		self.beta_ref = beta_tref
		self.t_stc = t_stc
		self.area = arr_area_m2

		self.climate_data = get_climate_data(self.latitude_deg, self.longitude_deg)
		self.index = self.climate_data.index

		self.delta = self.get_delta()
		self.beta = self.get_beta()
		self.r_pot_s = self.get_r_pot_s()
		self.cos_theta = self.get_cos_theta()
		self.m_theta = self.get_m_theta()
		self.t_max = self.get_max_transmittance()
		self.dt = self.get_dt()
		self.t_correction = self.get_transmittance_correction()
		self.r_pot = np.clip(mean_annual_over_days(self.r_pot_s), a_min=0, a_max=None)
		# daily indicent solar radiation
		self.rg = self.get_r_g()
		# daily outgoing longwave radiation
		self.r_nl = self.get_r_nl()

		# get place-specific metrics
		self.direct_solar_radiation = direct_solar_radiation(self.latitude_rad, self.longitude_rad)
		self.direct_solar_radiation_day_means = mean_nonzero_annual_over_days(self.direct_solar_radiation)

		self.day_means = compute_climate_data_doy_means(self.climate_data)
		self.annual_means = compute_climate_data_annual_means(self.climate_data)
		# extract datastreams
		self.temps = self.climate_data[CLIMATE_MODEL_METRICS[0]].to_numpy()
		self.cloud_cover = self.climate_data[CLIMATE_MODEL_METRICS[1]]
		self.cc_correction = self.get_cloud_cover_correction_factor()
		self.rel_humidity = self.climate_data[CLIMATE_MODEL_METRICS[2]]
		self.pressure = self.climate_data[CLIMATE_MODEL_METRICS[3]]
		# align ab initio direct rad data with real weather data
		self.direct_solar_radiation_aligned = np.concat([self.direct_solar_radiation_day_means[day - 2, None] for day in self.climate_data.index.day_of_year])

		# compute_properties
		self.efficiency: np.array = self.get_efficiency_timeseries()
		self.total_incident_radiation = self.direct_solar_radiation_aligned + self.r_nl
		self.expected_output = self.get_expexted_output()
		self.out_annual_mean = self.get_annual_means(self.expected_output)

		self.write_report()
		return

	def get_cloud_cover_correction_factor(self):
		return 1 - self.cloud_cover * CLOUD_TRANSMISSIVITY_AT_100 / 100

	def reindex_annualized_data(self, metric: np.array):
		return np.concat([metric[day - 2, None] for day in self.index.day_of_year])

	def get_daily_means_over_years(self, daily_values: np.array) -> np.array:

		out: list[float] = list()
		for day in range(365):
			idxer = (self.index.day_of_year == day + 1)
			on_this_day = daily_values[idxer]
			mean_on_this_day = np.mean(on_this_day)
			out.append(mean_on_this_day)

		arr_out = np.array(out)
		return arr_out

	def get_annual_means(self, daily_values: np.array) -> np.array:

		years = np.unique(self.index.year)
		out: list[float] = list()
		for year in years:
			idxer = (self.index.year == year)
			on_this_year = daily_values[idxer]
			mean_on_this_year = np.mean(on_this_year)
			out.append(mean_on_this_year)

		arr_out = np.array(out)
		return arr_out

	def get_r_pot_s(self):
		return SOLAR_CONSTANT * np.sin(self.beta)

	def get_delta(self):
		# solar elevation angle
		year_fraction_vector: np.array = np.linspace(0, 1, EVAL_PTS)
		# solar_axial_offset: np.array = (AXIAL_TILT * np.cos(tau * year_fraction_vector - self.latitude_rad))
		solar_axial_offset: np.array = (AXIAL_TILT * np.cos(tau * year_fraction_vector - self.latitude_rad - tau / 2))
		return solar_axial_offset

	def get_beta(self):
		return cos(self.latitude_rad) * np.cos(self.delta) + sin(self.latitude_rad) * np.sin(self.delta)

	def get_r_g(self):
		# daily incident radiation
		r_pot_all = self.reindex_annualized_data(self.r_pot)
		t_max_all = self.reindex_annualized_data(self.t_max)
		return r_pot_all * t_max_all * self.t_correction

	def get_r_pot_daily_mean(self):
		return mean_annual_over_days(self.r_pot_s)

	def get_m_theta(self):
		return 1 / self.cos_theta

	def get_max_transmittance(self):
		num = self.r_pot_s * (0.87 ** self.m_theta)
		num_sum = mean_annual_over_days(num)
		return num_sum / (mean_annual_over_days(self.r_pot_s))

	def get_dt(self):
		return self.climate_data["temperature_2m_max"] - self.climate_data["temperature_2m_min"]

	def get_transmittance_correction(self):
		b = 0.031 + 0.201 * np.exp(-0.185 * self.dt)
		c = 1.5
		return 1 - 0.9 * np.exp(-b * (self.dt ** c))

	def get_cos_theta(self):
		return sin(self.latitude_rad) * np.sin(self.delta) + cos(self.latitude_rad) * np.cos(self.delta)

	def get_r_nl(self):
		sigma = 4.903e-9  # MJ / K^-4 m^-2
		coeff = 0.34 - 0.14 * np.sqrt(self.climate_data["pressure_msl_mean"])
		t_max_4 = (self.climate_data["temperature_2m_max"] + 273.15)**4
		t_min_4 = (self.climate_data["temperature_2m_min"] + 273.15)**4
		coeff_2 = 0.5 * (t_max_4 + t_min_4)
		return sigma * coeff * coeff_2

	def get_efficiency_timeseries(self) -> np.array:
		eta_series: np.array = self.eta_ref * (1 - self.beta_ref * (self.temps - self.t_stc))
		return eta_series

	def get_expexted_output(self):
		return self.efficiency * self.total_incident_radiation * self.cc_correction

	def get_regression(self, timeseries: np.array) -> tuple[np.array, float, float, float]:
		slope, intercept, r2, p, std = linregress(np.arange(timeseries.size), timeseries)
		return slope * np.arange(timeseries.size) + intercept, slope, intercept, r2

	def make_figures(self):

		# day of year means
		fig, [[ax00, ax01], [ax10, ax11]] = plt.subplots(2, 2, figsize=(12, 9))
		if self.placename is not None:
			fig.suptitle(self.placename)

		self.plot_dir_radiation_profile(ax00)
		self.plot_efficiency(ax01)
		self.plot_annual_radiation(ax10)
		self.plot_annual_outputs(ax11)
		plt.show()

		fig, [[ax00, ax01], [ax10, ax11]] = plt.subplots(2, 2, figsize=(12, 9))
		if self.placename is not None:
			fig.suptitle(self.placename)

		self.plot_temperature_over_years(ax00)
		self.plot_cloud_cover_over_years(ax01)
		self.plot_humidity_over_years(ax11)
		self.plot_t_range_over_years(ax10)
		plt.show()

	def plot_dir_radiation_profile(self, ax: plt.Axes):

		day_of_year = np.linspace(0, 1, EVAL_PTS) * 365
		day_of_year_means = np.linspace(0, 1, 365) * 365
		ax.plot(day_of_year, self.direct_solar_radiation, label="Top of atmosphere", c=(1, 1, 0, 0.5))
		ax.plot(day_of_year_means, self.r_pot, label="Peak incident potential", c="r")
		ax.plot(day_of_year_means, self.direct_solar_radiation_day_means, label="Day mean", c="b")
		ax.plot(day_of_year_means, self.direct_solar_radiation_day_means * self.get_daily_means_over_years(self.cc_correction), label="Day mean w/ cloud effects", c="k")
		ax.legend()
		ax.set_xlabel("Day of year")
		ax.set_ylabel("P (W m$^{-2}$)")
		ax.set_title("Radiation profile")

		return

	def plot_efficiency(self, ax: plt.Axes):

		idx = self.index
		eta_annual_mean = self.get_annual_means(self.efficiency)
		ax.plot(idx, self.efficiency, label=r"$\eta$", c="k")
		ax.plot(idx[idx.day_of_year == 1], eta_annual_mean, label=r"Mean $\eta$", c="r")
		regression_line, m, b, r2 = self.get_regression(eta_annual_mean)
		ax.plot(idx[idx.day_of_year == 1], regression_line, label=f"Trend ({round(10000 * m, 3)}BP/year, $R^2$={round(r2, 3)})", c="b")
		ax.legend()
		ax.set_xlabel("Date")
		ax.set_ylabel("%")
		ax.set_title("Efficiency")

	def plot_annual_radiation(self, ax: plt.Axes):

		idx = self.index
		rad_annual_mean = self.get_annual_means(self.total_incident_radiation)
		ax.plot(idx, self.total_incident_radiation, label=r"$P_{in}$", c="k")
		ax.plot(idx[idx.day_of_year == 1], rad_annual_mean, label=r"Mean $P_{in}$", c="r")
		regression_line, m, b, r2 = self.get_regression(rad_annual_mean)
		ax.plot(idx[idx.day_of_year == 1], regression_line, label=f"Trend ({round(m, 3)}/year, $R^2$={round(r2, 3)})", c="b")
		ax.legend()
		ax.set_xlabel("Date")
		ax.set_ylabel("P$_{in}$ (W m$^{-2}$)")
		ax.set_title("Radiation")

		return

	def plot_annual_outputs(self, ax: plt.Axes):

		idx = self.index

		out_annual_mean = self.get_annual_means(self.expected_output)
		ax.plot(idx, np.clip(self.expected_output, a_min=0, a_max=None) * w_day_to_kwh_year, label=r"$P_{out}$", c="k")
		ax.plot(idx[idx.day_of_year == 1], out_annual_mean * w_day_to_kwh_year, label=r"Mean $P_{out}$", c="r")
		regression_line, m, b, r2 = self.get_regression(out_annual_mean * w_day_to_kwh_year)
		ax.plot(idx[idx.day_of_year == 1], regression_line, label=f"Trend ({round(m, 3)}/year, $R^2$={round(r2, 3)})", c="b")
		ax.legend()
		ax.set_xlabel("Date")
		ax.set_ylabel("P$_{out}$ (kWh m$^{-2}$)")
		ax.set_title("Radiation")

	def split_to_daily(self, arr: np.array) -> list[np.array]:

		included_years = np.unique(self.index.year)
		out: list[np.array] = list()

		for this_year in included_years:
			this_year_vals = arr[self.index.year == this_year]
			out.append(this_year_vals)

		return out

	def plot_temperature_over_years(self, ax: plt.Axes):

		annual_daily_temps = self.split_to_daily(self.temps)
		n_years = len(annual_daily_temps)

		for n in range(1, n_years - 1):
			annual_vals = annual_daily_temps[n]
			ax.plot(np.arange(np.size(annual_vals)), annual_vals, c=(n / n_years, 0, 1 - n / n_years))

		ax.set_title("Future temperature profiles")
		ax.set_xlabel("Day of year")
		ax.set_ylabel("°C")

		# plot first and last
		ax.plot(np.arange(np.size(annual_daily_temps[0])), annual_daily_temps[0], c=(1, 0.5, 0), label="2025")
		ax.plot(np.arange(np.size(annual_daily_temps[-1])), annual_daily_temps[-1], c=(1, 0.9, 0), label="2049")
		ax.legend()

		return

	def plot_cloud_cover_over_years(self, ax: plt.Axes):

		annual_daily_temps = self.split_to_daily(self.cloud_cover)
		n_years = len(annual_daily_temps)

		for n in range(1, n_years - 1):
			annual_vals = annual_daily_temps[n]
			ax.plot(np.arange(np.size(annual_vals)), annual_vals, c=(n / n_years, 0, 1 - n / n_years))

		ax.set_title("Future cloud cover profiles")
		ax.set_xlabel("Day of year")
		ax.set_ylabel("%")

		# plot first and last
		ax.plot(np.arange(np.size(annual_daily_temps[0])), annual_daily_temps[0], c=(1, 0.5, 0), label="2025")
		ax.plot(np.arange(np.size(annual_daily_temps[-1])), annual_daily_temps[-1], c=(1, 0.9, 0), label="2049")
		ax.legend()

		return

	def plot_humidity_over_years(self, ax: plt.Axes):

		annual_daily_temps = self.split_to_daily(self.rel_humidity)
		n_years = len(annual_daily_temps)

		for n in range(1, n_years - 1):
			annual_vals = annual_daily_temps[n]
			ax.plot(np.arange(np.size(annual_vals)), annual_vals, c=(n / n_years, 0, 1 - n / n_years))

		ax.set_title("Future humidity profiles")
		ax.set_xlabel("Day of year")
		ax.set_ylabel("%")

		# plot first and last
		ax.plot(np.arange(np.size(annual_daily_temps[0])), annual_daily_temps[0], c=(1, 0.5, 0), label="2025")
		ax.plot(np.arange(np.size(annual_daily_temps[-1])), annual_daily_temps[-1], c=(1, 0.9, 0), label="2049")
		ax.legend()

		return

	def plot_t_range_over_years(self, ax: plt.Axes):

		annual_daily_temps = self.split_to_daily(self.climate_data["temperature_2m_max"] - self.climate_data["temperature_2m_min"])
		n_years = len(annual_daily_temps)

		for n in range(1, n_years - 1):
			annual_vals = annual_daily_temps[n]
			ax.plot(np.arange(np.size(annual_vals)), annual_vals, c=(n / n_years, 0, 1 - n / n_years))

		ax.set_title("Future daily temp range profiles")
		ax.set_xlabel("Day of year")
		ax.set_ylabel(r"$\Delta$T (°C)")

		# plot first and last
		ax.plot(np.arange(np.size(annual_daily_temps[0])), annual_daily_temps[0], c=(1, 0.5, 0), label="2025")
		ax.plot(np.arange(np.size(annual_daily_temps[-1])), annual_daily_temps[-1], c=(1, 0.9, 0), label="2049")
		ax.legend()

	def write_report(self):
		out_2025, out_2049 = self.out_annual_mean[0] * w_day_to_kwh_year * self.area, self.out_annual_mean[-1] * w_day_to_kwh_year * self.area
		s0 = "reduction" if out_2025 > out_2049 else "increase"
		c0 = 1 - out_2025 / out_2049 if out_2025 < out_2049 else 1 - out_2049 / out_2025

		temp_series = self.get_annual_means(self.temps)
		temp_change = temp_series[-1] - temp_series[0]
		temp_trend, mt, bt, r2t = self.get_regression(temp_series)
		cloud_series = self.get_annual_means(self.cloud_cover)
		cloud_change = cloud_series[-1] - cloud_series[0]
		cloud_trend, mc, bc, r2c = self.get_regression(cloud_series)
		humid_series = self.get_annual_means(self.rel_humidity)
		humid_change = humid_series[-1] - humid_series[0]
		humid_trend, mh, bh, r2h = self.get_regression(humid_series)

		eff_ann = self.get_annual_means(self.efficiency)
		change_due_to_t = (eff_ann[-1] - eff_ann[0]) / eff_ann[0]
		cc_red_ann = self.get_annual_means(self.cc_correction)
		change_due_to_cc = (cc_red_ann[-1] - cc_red_ann[0]) / cc_red_ann[0]

		s1 = "rising" if temp_change > 0 else "falling"
		c1 = round(float(change_due_to_t) * 100, 1)
		s2 = "increasing" if cloud_change > 0 else "decreasing"
		c2 = round(float(change_due_to_cc) * 100, 1)

		report_out = (
			f"Solar array report for {self.placename} (Array properties: A = {self.area} m^2, η(STC) = {self.eta_ref})\n"
			f"\t2025 mean expected output: {round(out_2025, 1)} kWh | 2049 mean expected output: {round(out_2049, 1)} kWh ({round(100 * c0, 1)}% {s0})\n"
			f"\t{c1}% change due to {s1} temperature effects on efficiency\n"
			f"\t{c2}% change due to {s2} cloud cover effects on atmosphere-ground transmissivity\n"
			f"Location temperature trends:\n"
			f"\t2025 mean: {round(float(temp_series[0]), 1)} °C | 2049 mean: {round(float(temp_series[-1]), 1)} °C\n"
			f"\tLinear model: T(Y - 2025) = {round(mt, 3)}Y + {round(bt, 1)} (R2 = {round(r2t, 3)})\n"
			f"Location cloud cover trends:\n"
			f"\t2025 mean: {round(cloud_series[0])}% | 2049 mean: {round(cloud_series[-1])}%\n"
			f"\tLinear model: CC(Y - 2025) = {round(mc, 3)}Y + {round(bc)} (R2 = {round(r2c, 3)})\n"
			f"Location humidity trends:\n"
			f"\t2025 mean: {round(humid_series[0])}% | 2049 mean: {round(humid_series[-1])}%\n"
			f"\tLinear model: H(Y - 2025) = {round(mh, 3)}Y + {round(bh)} (R2 = {round(r2h, 3)})\n"
		)
		print(report_out)
		return


if __name__ == '__main__':

	places = ("Chapel Hill, NC", "Helsinki, Finland", "Patagonia, Argentina", "North Pole", )
	hk_lat = 22 + 21 / 60 + 9 / 3600
	hk_long = 114 + 8 / 60 + 21 / 3600
	sa = SolarArray(hk_lat, hk_long, placename="Hong Kong")
	sa.make_figures()
	for place in places:
		sa = SolarArray(*LOCATION_DICT.get(place), placename=place)
		sa.make_figures()
