import os
import sys

from pymarine.waves.wave_spectra import (spectrum_jonswap, d_omega_e_prime, omega_e_vs_omega)

from wavesimulator.utils import (clear_argument_list,
                                 get_logger, move_script_path_to_back_of_search_path)

# remove the script path from the path to avoid messing up the __version
move_script_path_to_back_of_search_path(__file__)

from scipy.constants import g as g0

from scipy import signal
from scipy.interpolate import interp1d
from numpy import pi

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import argparse
import logging

rcParams["font.size"] = 10


def max_value_from_dict(the_dict):
    max_value = None
    max_key = None
    for key, value in the_dict.iteritems():
        if max_value is None or value > max_value:
            max_value = value
            max_key = key

    return max_key, max_value


def parse_arguments():
    # clean the arguments list given on the command line in case you run from cygwin
    sys.argv = clear_argument_list(sys.argv)

    # parse the command line arguments
    parser = argparse.ArgumentParser(description="A script for creating plots out of the fatigue monitoring database",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--debug', help="Print lots of debugging statements", action="store_const",
                        dest="log_level", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="log_level",
                        const=logging.INFO,
                        default=logging.INFO)
    parser.add_argument('-q', '--quiet', help="Be quiet: no output", action="store_const", dest="log_level",
                        const=logging.WARNING, default=logging.INFO)
    parser.add_argument('monitor_file', type=str, help="Required argument: the fatigue database ")

    parser.add_argument('--use_log', help="USe log scale for PSD plot", action="store_true")

    parser.add_argument('--time_max', type=float, help="Maximum time value in time series")
    parser.add_argument('--f_max', type=float, help="Maximum frequency value in time series")
    parser.add_argument('--S_max', type=float, help="Maximum PSD value in time series")

    parser.add_argument('--nfft', type=int, help="Number of fft bins per block", default=256)

    # parse the command line arguments
    args = parser.parse_args()

    return args, parser


def main():
    args, parser = parse_arguments()

    # -----------------------------------------------------------------------
    # initialise the logging
    # -----------------------------------------------------------------------
    init_logging()
    log = get_logger(__name__)
    log.setLevel(args.log_level)

    log.info("Plotting monitoring data {}".format(args.monitor_file))
    log.debug("In debugging mode")

    # plot current working folder and script locationn
    log.debug("getcwd = {}   sys.argv[0] = {}".format(os.getcwd(), sys.argv[0]))

    # check if we can find the fatigue database
    if not os.path.exists(args.monitor_file):
        log.warning("Going out")
        parser.error("Can not find the monitoring file: {}".format(args.monitor_file))

    # analyse the fatigue database filename and location
    mon_db_path, mon_db_file_name = os.path.split(args.monitor_file)
    mon_db_base_name, mon_db_base_ext = os.path.splitext(mon_db_file_name)

    log.debug("path={} file_base = {} ext = {}".format(mon_db_path, mon_db_base_name, mon_db_base_ext))

    db = pd.read_excel(args.monitor_file)

    db.set_index("Time", drop=True, inplace=True)
    db.dropna(inplace=True)
    db.info()

    dt = np.diff(db.index.values)[0]
    ff = 1.0 / dt

    log.debug("Sampling frequency: {}".format(ff))

    # create the figure with the 6 subfigure which will contain the 6-DOF acceleration
    titles = list()
    figs = list()
    axes = list()
    col_names = list()
    y_labels = list()
    gs = list()
    y_label_base = list()

    titles.append("Time series")
    titles.append("Spectra series")
    y_label_base.append("Amplitude [m]")
    y_label_base.append("PSD")

    n_row = 5
    n_col = 1

    for cnt, title in enumerate(titles):
        figs.append(plt.figure(figsize=(12, 12)))
        gs.append(gridspec.GridSpec(n_row, n_col))
        # create 3x2 list with the indices pointing to the six DOF
        axes.append(list())
        col_names.append(list())
        y_labels.append(list())

        figs[-1].canvas.set_window_title(title=titles[cnt])
        gs[-1].update(left=0.1, right=0.98, hspace=0.3)

    i_row = 0
    j_col = 0

    # u_monitors = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    u_monitors = np.linspace(0, 5, 5, endpoint=False)
    log.debug(u_monitors)

    ylims = [[4, 4, 4, 5, 5],
             [8, 8, 8, 8, 8],
             ]

    cnt = 0
    for name in db.columns:
        u_mon = float(name)
        if u_mon not in u_monitors:
            continue

        log.debug("Adding column {} {}".format(cnt, name))

        for i_ax, ax in enumerate(axes):
            ax.append(figs[i_ax].add_subplot(gs[i_ax][i_row, j_col]))

            ax[-1].text(0.8, 1.02, "U = {} m/s".format(name), transform=ax[-1].transAxes, fontdict=dict(size=10))

            y_label = y_label_base[i_ax]
            log.debug(y_label)
            ax[-1].set_ylabel(y_label)

        # take care of the time series plot
        db.plot(y=[name], style='-b', ax=axes[0][-1], legend=None)
        if args.time_max is not None:
            axes[0][-1].set_xlim(0, args.time_max)
        axes[0][-1].set_ylim(-3, 3)

        # freq, psd = signal.periodogram(db[name], ff, nfft=args.nfft, window="parzen")
        freq, psd = signal.welch(db[name], ff, nfft=args.nfft, window="hanning")

        # express psd in omega
        psd_vs_omega = psd / (2 * np.pi)

        freq_omega = 2 * pi * freq

        if u_mon > 0:
            omega_critical = g0 / (2.0 * u_mon)
            omega_phase = g0 / u_mon
        else:
            omega_critical = 1e10000
            omega_phase = 1e10000

        omega_critical_e = omega_e_vs_omega(omega_critical, u_mon)
        omega_phase_e = omega_e_vs_omega(omega_phase, u_mon)

        js_spec_vs_omega = spectrum_jonswap(freq_omega, Hs=3, Tp=10)

        d_omega_E_d_omega = d_omega_e_prime(freq_omega, u_mon)
        # d_omega_E_d_omega_e = omega_e_vs_omega(d_omega_E_d_omega, u_mon)
        tiny = np.exp(-53)
        omega_e = omega_e_vs_omega(freq_omega, u_mon)
        d_omega_E_d_omega = np.where(abs(d_omega_E_d_omega) < tiny, np.sign(d_omega_E_d_omega) * tiny,
                                     d_omega_E_d_omega)

        js_spec_prime_vs_omega_e = js_spec_vs_omega / abs(d_omega_E_d_omega)

        d_freq1 = np.diff(freq)
        d_freq1 = np.append(d_freq1, np.array([d_freq1[-1]]))
        d_freq2 = np.diff(freq_omega)
        d_freq2 = np.append(d_freq2, np.array([d_freq2[-1]]))
        Hs1 = 4 * np.sqrt((js_spec_vs_omega * d_freq2).sum())
        Hs2 = 4 * np.sqrt((psd_vs_omega * d_freq2).sum())

        log.info("u={:10s} : Hs1={:.2f} Hs2={:.2f}".format(name, Hs1, Hs2))

        # two solutions of omega from omega_e
        omega_one = omega_critical * (1 - np.sqrt(2 * omega_e / omega_critical))
        omega_one_e = omega_e_vs_omega(omega_one, u_mon)  # + omega_critical_e
        omega_two = omega_critical * (1 + np.sqrt(2 * omega_e / omega_critical))
        omega_two_e = omega_e_vs_omega(omega_two, u_mon)  # + omega_critical_e

        js_spec_prime_vs_omega_one = np.where(d_omega_E_d_omega > 0, js_spec_prime_vs_omega_e, 0)
        js_spec_prime_vs_omega_two = np.where(d_omega_E_d_omega < 0, js_spec_prime_vs_omega_e, 0)

        omega_e_swap = np.where(d_omega_E_d_omega > 0, omega_e, omega_critical - omega_e)
        swap_index_list = np.where(np.diff(np.sign(d_omega_E_d_omega)))[0]
        if swap_index_list:
            index_critical = swap_index_list[0]
        else:
            index_critical = None

        js_spec_prime_vs_omega_e_1 = np.where(d_omega_E_d_omega > 0.0, js_spec_prime_vs_omega_e, 0)
        js_spec_prime_vs_omega_e_2 = np.where(d_omega_E_d_omega < 0.0, js_spec_prime_vs_omega_e, 0)
        js_spec_prime_vs_omega_e_folded = js_spec_prime_vs_omega_e
        js_spec_prime_vs_omega_e_clipped = js_spec_prime_vs_omega_e
        if index_critical is not None:
            n_js_size = js_spec_prime_vs_omega_e.size
            j1 = np.zeros(n_js_size)

            # add the points starting from the index 'index_critical' to the end of the array  (the folded freqencies)
            # to the starting of the array. There are two options: either n_part_1 > n_part_2, and n_part_1<n_part_2
            # both case should work

            n_part_1 = index_critical  # number of point up to index_critical (with including the index at index_critical)
            i_end_swap = min(index_critical + n_part_1 - 1, n_js_size - 1)  # the end index of the swapped part
            n_part_2 = i_end_swap - index_critical  # the number of point in the second part
            i_start_swap = max(0, index_critical - n_part_2)  # the index starting at the first part

            log.debug("n_js {}  n_st {} n_end {} n1 {} n2 {} : {}".format(n_js_size, i_start_swap, i_end_swap, n_part_1,
                                                                          n_part_2, index_critical))

            # copy the first part of the spectrum upto index_critical
            j1[: index_critical] = js_spec_prime_vs_omega_e[:index_critical]

            # add the frequencies at the second part of the spectrum which is folded back
            j1[i_start_swap: index_critical] += js_spec_prime_vs_omega_e[i_end_swap:index_critical:-1]

            # copy the results to a new array folded
            js_spec_prime_vs_omega_e_folded = j1

            # also as second option: just set the second folced part to zero
            js_spec_prime_vs_omega_e_clipped[index_critical:] = 0

        d_freq3 = np.diff(omega_e_swap)
        d_freq3 = np.append(d_freq3, np.array([d_freq3[-1]]))
        Hs3 = 4 * np.sqrt(np.nansum(js_spec_prime_vs_omega_e * d_freq3))

        # cum_sum_energy_miss = abs(np.nan_to_num(js_spec_prime_vs_omega_e) * d_freq3).cumsum()
        cum_sum_energy = abs(np.nan_to_num(js_spec_prime_vs_omega_e_folded) * abs(d_freq3)).cumsum()
        # omega_e_new = np.linspace(0, freq_omega[-1], 100, endpoint=True)
        omega_e_new = freq_omega
        d_omega_e_new = np.diff(omega_e_new)[0]

        f_inter = interp1d(omega_e_swap, cum_sum_energy, bounds_error=False, fill_value="extrapolate")
        cum_sum_energy_new = np.nan_to_num(f_inter(omega_e_new))

        # f_inter = interp1d(omega_e_swap, cum_sum_energy_miss, bounds_error=False, fill_value=0)
        # cum_sum_energy_new_miss = f_inter(omega_e_new)

        js_tmp = np.diff(cum_sum_energy_new) / d_omega_e_new
        js_spec_prime_vs_omega_e_int = np.append(js_tmp, np.array([js_tmp[-1]]))

        # to try the non-folded version: no difference
        # js_spec_prime_vs_omega_e_int_miss = np.diff(cum_sum_energy_new_miss) / d_omega_e_new
        # js_spec_prime_vs_omega_e_int_miss = np.append(js_spec_prime_vs_omega_e_int_miss, np.array([js_spec_prime_vs_omega_e_int_miss[-1]]))

        Hs4 = 4 * np.sqrt((js_spec_prime_vs_omega_e_int * d_omega_e_new).sum())
        log.debug("hier Hs/4 and last {} {}".format(np.power(Hs4 / 4., 2), js_spec_prime_vs_omega_e_int.max()))

        axes[1][-1].text(0.1, 1.02,
                         u"Hs$_{JS}$=" + "{:.2f} m".format(Hs1) + u"  Hs$_{Meas}$=" + "{: .2f} / {:.2f}".format(Hs2,
                                                                                                                Hs4),
                         transform=axes[1][-1].transAxes, fontdict=dict(size=10))

        axes[1][-1].plot(freq_omega, psd_vs_omega, "-r", label="Measured")
        axes[1][-1].plot(freq_omega, js_spec_vs_omega, '-g', label="Jonswap")
        # axes[1][-1].plot(omega_e, js_spec_prime_vs_omega_e, '-b', label="Jonswap_E1")
        # axes[1][-1].plot(omega_e_swap, js_spec_prime_vs_omega_e_clipped, '--c', label="Jonswap_E2", linewidth=3)
        axes[1][-1].plot(omega_e_new, js_spec_prime_vs_omega_e_int, '-b', label="Jonswap_E2", linewidth=2)
        # axes[1][-1].plot(omega_e_new, js_spec_prime_vs_omega_e_int_miss, '--y', label="Jonswap_E2", linewidth=2)
        # axes[1][-1].plot(omega_e_swap, cum_sum_energy, '-oc', label="cumsum1", linewidth=2)
        # axes[1][-1].plot(omega_e_new, cum_sum_energy_new, '-oy', label="cumsum2", linewidth=2)
        # axes[1][-1].plot(omega_e, js_spec_prime_vs_omega_e_folded, '--c', label="Jonswap_Folded")
        # axes[1][-1].plot(omega_e, js_spec_prime_vs_omega_e_clipped, '--k', label="Jonswap_clipped")
        # axes[1][-1].plot(omega_e, js_spec_prime_vs_omega_e_1, '-b', label="Jonswap_E1")
        # axes[1][-1].plot(omega_e, js_spec_prime_vs_omega_e_2, '-c', label="Jonswap_E2")

        # axes[1][-1].plot(omega_two_e , js_spec_prime_vs_omega_two, '--c', label="Jonswap_E1 ")
        # axes[1][-1].plot(omega_e, js_spec_prime_vs_omega_two, '--k', label="Jonswap_E1 ")
        # axes[1][-1].plot(omega_one_e, js_spec_prime_vs_omega_two, '--y', label="Jonswap_E1 ")
        # axes[1][-1].plot(freq_omega, omega_e, '--k', label="omega_e ")
        # axes[1][-1].plot(freq_omega, omega_e_swap, '--y', label="omega_e ")
        # axes[1][-1].plot([omega_critical_e, omega_critical_e], [0, 100], '--k')
        # axes[1][-1].plot([omega_phase_e, omega_phase_e], [0, 100], '--y')
        # axes[1][-1].plot(omega_two, abs(js_spec_prime_vs_omega_e), '--y', label="Jonswap_E 2")

        if args.use_log:
            S_min = 1e-6
            axes[1][-1].semilogy()
        else:
            S_min = 0
        if args.f_max is not None:
            axes[1][-1].set_xlim(0, args.f_max)
        if args.S_max is not None:
            axes[1][-1].set_ylim(S_min, args.S_max)
        else:
            axes[1][-1].set_ylim(S_min, ylims[j_col][i_row])
        axes[1][-1].legend(loc="best")
        # axes[1][-1].set_ylim(1e-8,100)

        if cnt > 0:
            axes[1][-1].legend().set_visible(False)

        if i_row == n_row - 1:
            axes[0][-1].set_xlabel("Time s")
            axes[1][-1].set_xlabel("$\omega$ rad/s")
        else:
            axes[0][-1].set_xlabel("")
            axes[1][-1].set_xlabel("")

        i_row += 1
        if i_row % n_row == 0:
            i_row = 0
            j_col += 1

        cnt += 1

    plt.show()


if __name__ == "__main__":
    main()
