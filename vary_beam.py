# importing required modules and packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# setting the runtime configuration parameters
# for desried formatting of matplotlib graphs and figures
mpl.rc(
    "axes",
    lw=2,
    labelsize=20,
    labelweight="bold",
    titleweight="bold",
    titlesize=20,
    titlepad=0,
)
mpl.rc("font", weight="bold", family="Times New Roman")
mpl.rc("xtick", direction="in", labelsize=18, top=True)
mpl.rc("ytick", direction="in", labelsize=18, right=True)
mpl.rc(["xtick.major", "ytick.major"], width=2.3, size=6)
mpl.rc(["xtick.minor", "ytick.minor"], visible=True, width=1.5, size=4)
mpl.rc("lines", linewidth=2.0)
mpl.rc(
    "legend",
    borderaxespad=1.0,
    borderpad=0.6,
    frameon=False,
    fancybox=False,
    fontsize=13,
)
mpl.rc("figure", autolayout=True)

# define efficiency fitted function for different gaps
# In my case, they are double exponential decay functions
def eff_fitfunc(x, p):
    return p[0] + p[1] * np.exp(-x / p[2]) + p[3] * np.exp(-x / p[4])


# fitting parameters of efficiency curves
# for first gap of 1 cm
gapft1 = np.array((0.00646, 94.60013, 13.49975, 0.06192, 317.82283))
# for second gap of 1 cm
gapft2 = np.array((0.00513, 0.10563, 51.94212, 0.05195, 305.48763))
# for third gap of 3 cm
gapft3 = np.array((0.00375, 0.0443, 84.21148, 0.03622, 334.99808))

# function for getting efficiency based on gap number and gamma energy
def gapbsd_effcncy(row):
    if row["gap"] == 1:
        return eff_fitfunc(row["E_gamma (keV)"], gapft1)
    elif row["gap"] == 2:
        return eff_fitfunc(row["E_gamma (keV)"], gapft2)
    elif row["gap"] == 3:
        return eff_fitfunc(row["E_gamma (keV)"], gapft3)
    return None


# storing the data in the gamma spectra data file in dataframe variable, df1
df1 = pd.read_csv("gamma_spectrum_data_and_other_properties.csv")

# add efficiency column in df1 corresponding to the gap of the recording and the gamma-energy
df1["efficiency"] = df1.apply(gapbsd_effcncy, axis=1)

# storing the df1 columns in simple variable names
area = df1["gamma_peak_area"]
effcncy = df1["efficiency"]
pulsr_area = df1["pulser_peak_area"]
t_c = df1["count_time"]
t_l = df1["lapse_time"]
# storing gamma emission probability in simple variable theta
theta = df1["gamma_emission_probability(%)"] / 100
# storing mass thickness in gm/cm^2 which is given in mg/cm^2
# in simple variable mass_thckns
mass_thckns = df1.loc[0, "foil_thickness(mg/cm^2)"] * 10 ** (-3)
# target mass number
targt_A = df1.loc[0, "target_A"]
# expected half-life
t_half = df1["expected_half_life(s)"].iloc[0]
# system (projectile + target)
system = df1["System"].iloc[0]
# beam energy
energy = df1["Beam_Energy(MeV)"].iloc[0]
# reaction channel
channel = df1["Channel"].iloc[0]
# expected decay constant
dcy_constnt = np.log(2) / t_half
# dead time correction factor employing the pulser used
plsr_corr = 50 * t_c / pulsr_area
# define # decays as panda series
decays = area * plsr_corr / (effcncy * theta)
# add the activity column in df1 with counting correction
df1["activity"] = decays * dcy_constnt * t_c / (t_c * (1 - np.exp(-dcy_constnt * t_c)))
# storing the defined activities from df1 columns in simple variable name activity
activity = df1["activity"]

# exponential decay model function to be fitted to the activities
def f(x, *p):
    return p[0] * np.exp(-np.log(2) * x / p[1])


# initial guess for the first fitting parameter of exponential decay fit
# which has to be the initial activity or activity just at the stop of irradiation
# i.e. t_l = 0
a_0 = activity.iloc[0] * np.exp(t_l.iloc[0] * dcy_constnt)
# initial guess for second parameter is equal to the half-life which is t_half

# defining the figure and axis for plotting the decay curve
fig1, ax1 = plt.subplots()
# plot t_l vs activity
ax1.scatter(t_l, activity, label="Decay rate")

# fit the exponential decay fit for activity and
# get the fitted parameters and the covariance matrix
popt, *_ = curve_fit(f, t_l, activity, (a_0, t_half))

# the obtained decay constant from the fitting
obtnd_dcycnst = np.log(2) / popt[1]

# plotting of the exponential decay fit
# define the x range for exponential decay fit function in which to be plotted
xFit = np.arange(0, 1.15 * t_l.max(), 1)
# plot the fit on the graph which contains the activities vs t_l
ax1.plot(
    xFit,
    f(xFit, popt[0], popt[1]),
    label=f"{popt[0]:.0f}exp(-ln(2)t/{popt[1]:.0f}s)",
)

# custominzing the decay curve
ax1.set_yscale("log")
# ax1.set_title()
ax1.set_xlim(left=0)
ax1.set_ylabel("Acivity (dps)")
ax1.set_xlabel("Time (s)")
ax1.legend()

# saving the decay curve
plt.savefig(f"decay_curve_{system}_{channel}_{energy}_MeV.png", dpi=600)

# storing the target properties and the beam properties df's
# in the variables df2
df2 = pd.read_csv("beam_current_vs_time.csv")

# concatenating the date and time columns as strings
df2["datetime"] = df2["date"].astype(str) + " " + df2["time"].astype(str)

# convertng the 'datetime' column to datetime type
df2["datetime"] = pd.to_datetime(df2["datetime"])

# calculating the time spent from start of the beam in seconds
df2["time_from_zero(s)"] = (df2["datetime"] - df2.loc[0, "datetime"]).dt.total_seconds()

# time intervals between readings of the current
df2["time_intervals"] = -df2["time_from_zero(s)"].diff(periods=-1)
df2["time_intervals"] = df2["time_intervals"].fillna(0)

# beam chargestate
bm_chrgstate = df2.loc[0, "beam_charge_state(+)"]

# conversion factor for converting beam current from amperes to particle per second
amptopA = 1 / (bm_chrgstate * constants.e)
# converting the current from nA to pps
df2["current(pps)"] = df2["current(nA)"] * constants.nano * amptopA
# converting the current from nA to pnA
df2["current(pnA)"] = df2["current(nA)"] / bm_chrgstate

# shifted time intervals column
df2["shifted_time_intervals"] = df2["time_intervals"].shift(1)
df2["shifted_time_intervals"] = df2["shifted_time_intervals"].fillna(0)

# shifted current column
df2["shifted_current(pps)"] = df2["current(pps)"].shift(1)
df2["shifted_current(pps)"] = df2["shifted_current(pps)"].fillna(0)

# reading irradiation time
irrad_time = df2["time_from_zero(s)"].iloc[-1]

# multiplying the current and time intervals column and summing
# gives currenttime in particle number units
currenttime_sump = (df2["time_intervals"] * df2["current(pps)"]).sum()
# gives current in pnA.s units
currenttime_sumpnAs = (df2["time_intervals"] * df2["current(pnA)"]).sum()
# defining the weighted average beam current
# in the units of pps
avrg_crntpps = (currenttime_sump) / (irrad_time)
# in the units of pnA
avrg_crntpnA = (currenttime_sumpnAs) / irrad_time

# converting some columns from df2 into numpy arrays
# for the vector calculations
tfrmz = df2["time_from_zero(s)"].to_numpy()
ntmitrvls = df2["shifted_time_intervals"].to_numpy()
ncrntpps = df2["shifted_current(pps)"].to_numpy()

## calculating the summation factors at the times for which the beam current changed or is read
# during the irradiation for the cross-section calculation with fluctuating beam

# initiating the np array for storing the factors as zero filled numpy array
myvar = np.zeros(tfrmz.shape)

# populating the np array
for i in np.arange(len(df2["time_intervals"])):
    for j in np.arange(i + 1):
        myvar[i] += (
            (1 - np.exp(-obtnd_dcycnst * ntmitrvls[j]))
            * (np.exp(-obtnd_dcycnst * (tfrmz[i] - tfrmz[j])))
            * ncrntpps[j]
        )
# storing the factor array as column of current df
df2["sum_factor"] = pd.Series(myvar)

# calculating the averaged beam current cross-section
# numerator for individual activity cross-section
xsctn_numind = activity * np.exp(obtnd_dcycnst * t_l) * targt_A
# numerator for activity at t_l = 0 or end of irradiation cross-section
xsctn_numA0 = popt[0] * targt_A
# denominator for cross-section
xsctn_den = (
    (1 - np.exp(-obtnd_dcycnst * irrad_time))
    * constants.N_A
    * mass_thckns
    * avrg_crntpps
)
# cross-section in mb
avrgcrnt_xsctionA0 = xsctn_numA0 * 10 ** (27) / xsctn_den
df1["avrgcrnt_xsctionA0(mb)"] = ""
df1.loc[0, "avrgcrnt_xsctionA0(mb)"] = avrgcrnt_xsctionA0
df1["avgcrntxsctn_ind(mb)"] = xsctn_numind * 10 ** (27) / xsctn_den

# calculating the exact fluctuating beam current cross-section
fact_sum = df2["sum_factor"].iloc[-1]
# cross-section in mb
flct_xsctnA0 = popt[0] * targt_A * 10**27 / (constants.N_A * fact_sum * mass_thckns)
df1["exctflctng_xsctnA0(mb)"] = ""
df1.loc[0, "exctflctng_xsctnA0(mb)"] = flct_xsctnA0
df1["exctflctng_xsctnind(mb)"] = (
    flct_xsctnA0 * activity * np.exp(obtnd_dcycnst * t_l) / popt[0]
)

# saving the cross-section calculations to a csv file which opens nicely
# in excel or libreoffice calc
df1.to_csv(f"{channel}@{system}@{energy}MeV_cross_section_calculation.csv")

## plotting the beam current as a function of time
# defining the figure and axis for plotting the beam current vs time
fig2, ax2 = plt.subplots()
# plotting the beam current by time in units of pnA
ax2.step(
    df2["time_from_zero(s)"],
    df2["current(pnA)"],
    where="post",
    label="Beam Current by time",
)

# plotting the weigted average beam current in units of pnA
ax2.hlines(
    y=avrg_crntpnA,
    xmin=0,
    xmax=df2["time_from_zero(s)"].iloc[-1],
    linestyle="dashed",
    label="Beam Current Weighted Average",
)
ax2.legend()
ax2.set_xlabel("Irradiation time (s)")
ax2.set_ylabel("Current (pps)")
# ax2.set_title(f'{system}@{energy}MeV')
plt.savefig(f"beam_current_by_time_{system}_{channel}_{energy}_MeV.png", dpi=600)

# adding # of live nuclei during irradiation time to df2
# average beam current live number numerator
navg_num = (
    flct_xsctnA0
    * 10 ** (-27)
    * avrg_crntpps
    * mass_thckns
    * constants.N_A
    * (1 - np.exp(-obtnd_dcycnst * tfrmz))
)
# exact beam current (fluctuating beam current) live number numerator
nexct_num = myvar * flct_xsctnA0 * 10 ** (-27) * mass_thckns * constants.N_A
n_den = targt_A * obtnd_dcycnst
df2["n_avg"] = pd.Series(navg_num / n_den)
df2["n_flct"] = pd.Series(nexct_num / n_den)

# converting fluctuating # of live nuclei pd Series to a numpy array
nflct = df2["n_flct"].to_numpy()

# saving the current and # live nuclei calculations to a csv file
df2.to_csv(f"{channel}@{system}@{energy}MeV_beam_current_vs_time.csv")

# creating dense x and y values for cubic spline plotting of
# live number of nuclei vs time for fluctuating beam current
f = interp1d(tfrmz, nflct, kind="cubic")
x_dense = np.linspace(0, irrad_time, 7000)
y_dense = f(x_dense)

# plotting the growth of # of live nuclei vs time during the irradiation time and
# the weigted average beam current and beam current by time

# defining the figure and axis for plotting the beam current vs time and
# live # of nuclei vs time during the irradiation time
fig3, ax3 = plt.subplots()
# setting color for ax3 axis
color1 = "blue"
# plotting varying beam current in units of pnA on ax3 axis
ax3.step(
    tfrmz,
    df2["current(pnA)"],
    where="post",
    color=color1,
    label="Beam Current by time",
    linestyle="solid",
)
# plotting the averaged-out beam current in units of pnA on ax3 axis
ax3.hlines(
    y=avrg_crntpnA,
    xmin=0,
    xmax=df2["time_from_zero(s)"].iloc[-1],
    linestyle="dashdot",
    color=color1,
    label="Beam Current Weighted Average",
)
# customizing ax3 axis
ax3.set_ylabel("Current (pnA)")
ax3.yaxis.label.set_color(color1)
ax3.set_xlabel("Irradiation time (s)")
ax3.tick_params(axis="y", which="both", color=color1, labelcolor=color1)
# loc argument is for the position of the legend on the graph of live number of nuclei
# for my graph these coordinates work better
# you will have to give different coordinates for getting right position of the legend on your graph 
ax3leg = ax3.legend(loc=(0.29, 0.65), prop={"weight": "bold"})
for text in ax3leg.get_texts():
    text.set_color(color1)
# creating twin axis ax4 for plotting the growth of live nuclei number
ax4 = ax3.twinx()
# setting color for ax4
color2 = "red"
# plotting the growth of live nuclei # using actual varying beam current on the twin axis (a4)
ax4.plot(
    x_dense,
    y_dense,
    label="Exact No. of live nuclei",
    color=color2,
    linestyle="dashed",
)
# plotting the growth of live nuclei # using averaged-out beam current on the twin axis (a4)
ax4.plot(
    tfrmz,
    df2["n_avg"],
    linestyle="dotted",
    lw=3,
    color=color2,
    label="Average No. of live nuclei",
)
# customizing ax4 axis
ax4.set_ylabel("No. of live nuclei")
ax4.yaxis.label.set_color(color2)
# loc argument is for the position of the legend on the graph of live number of nuclei
# for my graph these coordinates work better
# you will have to give different coordinates for getting right position of the legend on your graph 
ax4leg = ax4.legend(loc=(0.29, 0.79), prop={"weight": "bold"})
ax4.yaxis.label.set_color(color2)
ax4.spines["left"].set_color(color1)
ax4.spines["right"].set_color(color2)
ax4.tick_params(axis="y", which="both", color=color2, labelcolor=color2)
for text in ax4leg.get_texts():
    text.set_color(color2)

plt.tight_layout()
plt.savefig(
    f"live_number_during_irradiation_time_{system}_{channel}_{energy}_MeV.png", dpi=600
)

# print statement if the program executes properly
print("Done! Created csv output files and png graphs in your python script folder")
