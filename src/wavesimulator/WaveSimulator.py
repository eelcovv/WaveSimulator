__author__ = "Eelco van Vliet"
__copyright__ = "Eelco van Vliet"
__license__ = "MIT"


import argparse
import logging
import sys
from os.path import basename, dirname, splitext
from re import search, sub

import pandas as pd
# import the wave modules
import pymarine.waves.wave_fields as wf
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtCore import QSettings, QByteArray, QTimer
from PyQt5.QtGui import QKeySequence, QIcon
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QWidget, QAction, QSplitter, QLabel, QFrame, QApplication
from numpy import radians, linspace, mod, full, nan, zeros, where, array
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import ParameterTree
from scipy import interpolate

# import the dialog for the spectra
import SpectrumPlotDlg
import SurfacePlotDlg
# import all the icon images (should be generated with
# pyrcc4 -py3 -o resources.py resources.qrc
# where resource.qrc is an xml file containing a list with the aliases
import resources
# import the parameter tree
from parameters import JSParameters
# import module for logging
# initiales my logging settings
from wavesimulator.utils import create_logger, clear_argument_list, get_logger
from wavesimulator.utils import get_parameter_list_key

# version control to be set up later
from wavesimulator import __version__
res_file = (
    resources.__file__
)  # this line is only to prevent the import resources statement to be


# removed by PyCharm


class WaveSimulatorGUI(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.logger = get_logger(__name__)
        self.logger.info("Start Jonswap time example")

        self.dirty = False
        self.filename = None
        self.exportfile = None
        self.moviefilebase = None

        self.setupGUI()

        self.monitor_waves = False

        # quick hack to add monitor points with different velocities (in m/s) along the positive x
        # direction
        if self.monitor_waves:
            self.wave_monitor_velocities = linspace(0, 20, 40, endpoint=False)
            self.wave_monitor_positions = zeros(40)

        self.wave_monitors = []

        # initialise the wave field used for the full transform over the whole k domain
        self.waves1D = []
        self.waves2D = []
        for i in range(2):
            self.waves1D.append(wf.Wave1D())
            self.waves2D.append(wf.Wave2D(self.waves1D[-1]))
            if self.monitor_waves:
                self.wave_monitors.append(None)

        self.pars = JSParameters()
        self.pars.connect_signals_to_slots()

        self.spectrumPlotDialog = None

        self.surfacePlotDialog = None

        self.playing = False
        self.repeat = False
        self.reset_time = False

        self.tree.setParameters(self.pars.Parameters, showTop=False)

        # call the method which copies all the parameters set in the parametertree to the wave
        # classes (1d and 2d)
        self.transfer_parameters()

        # create the dialog without showing it such that we can connect it to a signal
        # also, pass the showSpectraButton such that we can uncheck it once the dialog is closed by
        # the close button
        self.spectrumPlotDialog = SpectrumPlotDlg.SpectrumPlotDlg(
            self.waves2D, self.showSpectraPlot
        )

        self.surfacePlotDialog = SurfacePlotDlg.SurfacePlotDlg(
            self.waves2D, self.showSurfacePlot, self.moviefilebase
        )

        # self.pars.Parameters.tree_parameter_changed(self.transfer_parameters)

        settings = QSettings()
        self.recentFiles = settings.value("RecentFiles") or []

        self.restoreGeometry(
            settings.value("WaveSimulatorMainWindow/Geometry", QByteArray())
        )

        self.setWindowTitle("Wave Simulator")

        with pg.BusyCursor():
            QTimer.singleShot(0, self.loadInitialFile)

    def transfer_parameters(self, parameter=None):
        """
        Method used to transfer the parameters from the pyqtgraph Parameters object to the
        WaveSimulator settings

        Parameters
        ----------
        parameter: str, optional
            Name of the section of the parameter tree we received a signal from. Default = None
        """

        if parameter is not None:
            self.logger.debug(
                "received signal from {} {}".format(parameter, type(parameter))
            )
            self.dirty = True

        parSpectrum = self.pars.Parameters.names["Jonswap"]
        parDomain = self.pars.Parameters.names["Domain"]
        parTime = self.pars.Parameters.names["Time"]

        for i, jswave2D in enumerate(self.waves2D):
            jswave = jswave2D.wave1D
            parWave = self.pars.Parameters.names["Wave {}".format(i + 1)]
            jswave.twoD = bool(parSpectrum.twoD.value())
            jswave.Hs = parSpectrum.Hs.value()
            jswave.Tp = parSpectrum.Tp.value()
            jswave.sigma = parSpectrum.sigma_gauss.value()
            jswave.spectrum_type = parSpectrum.spectrum_type.value()
            jswave.spectrum_version = parSpectrum.spectrum_version.value()
            jswave.gamma = parSpectrum.gamma.value()

            # only waves from 0 to 180 allowed (0 is north an 90 is east and 180 is south).
            jswave2D.Theta_0 = radians(parSpectrum.theta0.value())
            jswave2D.Theta_s_spreading_factor = (
                parSpectrum.theta_s_spreading_factor.value()
            )
            jswave2D.theta_area_fraction = parWave.theta_area_percentage.value() / 100.0
            jswave2D.n_theta_nodes = int(parDomain.n_theta_nodes.value())

            jswave2D.xmax = parDomain.xmax.value()
            jswave2D.xmin = parDomain.xmin.value()
            jswave2D.Lx = parDomain.Lx.value()
            jswave2D.nx_points = parDomain.nx.value()

            jswave2D.kx_min = parDomain.kx_min.value()
            jswave2D.kx_max = parDomain.kx_max.value()
            jswave2D.delta_kx = parDomain.delta_kx.value()
            jswave2D.n_kx_nodes = int(parDomain.n_kx_nodes.value())

            jswave2D.ymay = parDomain.ymax.value()
            jswave2D.ymin = parDomain.ymin.value()
            jswave2D.Ly = parDomain.Ly.value()
            jswave2D.ny_points = parDomain.ny.value()

            jswave2D.ky_min = parDomain.ky_min.value()
            jswave2D.ky_may = parDomain.ky_max.value()
            jswave2D.delta_ky = parDomain.delta_ky.value()
            jswave2D.n_ky_nodes = int(parDomain.n_ky_nodes.value())

            jswave.xmax = parDomain.xmax.value()
            jswave.xmin = parDomain.xmin.value()
            jswave.Lx = parDomain.Lx.value()
            jswave.nx_points = parDomain.nx.value()

            jswave.kx_min = parDomain.kx_min.value()
            jswave.kx_max = parDomain.kx_max.value()
            jswave.delta_kx = parDomain.delta_kx.value()
            jswave.n_kx_nodes = int(parDomain.n_kx_nodes.value())

            jswave.t_start = parTime.t_start.value()
            jswave.t_end = parTime.t_end.value()
            jswave.t_length = parTime.duration.value()
            jswave.nt_samples = int(parTime.n_time.value())
            jswave.delta_t = parTime.delta_t.value()
            jswave.time = parTime.current_time.value()
            jswave.t_index = parTime.current_index.value()

            jswave.wave_construction = get_parameter_list_key(parWave.wave_construction)
            if jswave.wave_construction == "DFTpolar":
                # a DFT algoritm is used, so do not mirror the spectrum
                jswave.mirror = False
            elif jswave.wave_construction in ("DFTcartesian", "FFT"):
                # for the FFT the spectral components should be mirrored around the N/2 axis
                # the same applies for the DFTcartesian
                jswave.mirror = True
            else:
                raise AssertionError(
                    "wave construction should be one of the following: DFTpolar, DFTcartesian, FFT"
                    ". Found {}. Something is wrong".format(jswave.wave_construction)
                )

            if jswave.seed is not int(parWave.seed.value()):
                jswave.seed = int(parWave.seed.value())
                jswave.update_phase = True

            jswave.picked_wave_index = int(parWave.picked_wave_index.value())

            jswave.sample_every = int(parWave.sample_every.value())
            jswave.E_limit_low = parWave.E_limit_low.value() / 100.0
            jswave.E_limit_high = parWave.E_limit_high.value() / 100.0

            # controls which wave are plotted
            jswave.plot.show = bool(parWave.update_and_show.value())
            jswave.plot.save_image = bool(parWave.save_movie_frames.value())
            jswave.plot.save_image = bool(parWave.save_movie_frames.value())
            jswave.plot.save_data = bool(parWave.save_amplitude_data.value())
            jswave.plot.scattersize = int(parWave.scatter_points.value())
            jswave.plot.linewidth = int(parWave.width.value())

            # set the color of the line to  plot this wave
            jswave.plot.color = parWave.color.value()

            # determine how to construct the wave (full range, subrange, or equal energy)
            # note that for the FFT solver, only the full range is allowed
            jswave.wave_selection = get_parameter_list_key(parWave.wave_selection)

            jswave.n_bins_equal_energy = int(parWave.n_bins_equal_energy.value())
            jswave.lock_nodes_to_wave_one = bool(parWave.lock_nodes_to_wave_one.value())
            jswave.use_subrange_energy_limits = bool(
                parWave.use_subrange_energy_limits.value()
            )

            if parameter is None or (not bool(search("Time", parameter))):
                # do not update the mesh if you are only updating the time value
                jswave.update_x_k_t_sample_space()
                self.logger.debug("calling calculate wave {}".format(jswave.name))
                jswave.calculate_spectra_modulus()

                if jswave.twoD and jswave.plot.show:
                    jswave2D.update_x_k_theta_sample_space()
                    jswave2D.calculate_spreading_function()
                    jswave2D.calculate_spectral_components()

                if self.monitor_waves:
                    column_names = [
                        "{:.1f}".format(vel) for vel in self.wave_monitor_velocities
                    ]
                    self.wave_monitors[i] = pd.DataFrame(
                        full((jswave.nt_samples, len(column_names)), nan),
                        columns=column_names,
                        index=linspace(
                            jswave.t_start,
                            jswave.t_end,
                            jswave.nt_samples,
                            endpoint=False,
                        ),
                    )

                    self.wave_monitors[i].index.name = "Time"

        if parameter is None or (not bool(search("Time", parameter))):
            if self.surfacePlotDialog:
                self.surfacePlotDialog.initSurfaces()

        self.updatePlots()
        if self.spectrumPlotDialog:
            self.spectrumPlotDialog.updatePlots()

    def stopMovie(self):
        self.reset_time = True
        self.playMovieAction.setChecked(False)
        self.playing = False

    def pauseMovie(self):
        self.reset_time = False
        self.playMovieAction.setChecked(False)
        self.playing = False

    def loopMovie(self):
        if self.loopMovieAction.isChecked():
            self.logger.info("switch loop movie ON")
            self.repeat = True
        else:
            self.logger.info("switch loop movie OFF")
            self.repeat = False

    def startMovie(self):
        self.playing = True
        self.stopMovieAction.setChecked(False)
        self.pauseMovieAction.setChecked(False)
        self.playMovie()

    def playMovie(self):
        self.logger.debug("start playing movie at {:.2f}".format(self.waves1D[0].time))

        while self.waves1D[0].time < self.waves1D[0].t_end or self.repeat:
            if self.waves1D[0].time >= self.waves1D[0].t_end:
                self.logger.info("reset time to zero")
                for wave in self.waves1D:
                    wave.time = 0.0
                    wave.t_index = 0

            # check if the stop button has been pressed
            if not self.playing:
                if self.reset_time:
                    for wave in self.waves1D:
                        wave.time = 0.0
                        wave.t_index = 0

                # draw last time with current time
                self.pars.Parameters.names["Time"].current_time.setValue(
                    self.waves1D[0].time
                )
                self.updatePlots()
                self.logger.debug("stop playing at {} s".format(self.waves1D[0].time))
                break

            # update the time step for both waves
            for i_wave, wave in enumerate(self.waves1D):
                wave.time += wave.delta_t
                wave.t_index += 1

            self.logger.debug(
                "Updating time step to {:.2f}".format(self.waves1D[0].time)
            )

            self.updatePlots()

            # wait until the plot is drawn, otherwise the gui is blocked
            QApplication.processEvents()

    def CloseSpectrumPlot(self):
        self.spectrumPlotDialog = None
        if self.showSpectraPlot.isChecked():
            self.showSpectraPlot.setChecked(False)

    def CloseSurfacePlot(self):
        self.surfacePlotDialog = None
        if self.showSurfacePlot.isChecked():
            self.showSurfacePlot.setChecked(False)

    def OpenSpectrumPlot(self):
        if self.showSpectraPlot.isChecked():
            self.spectrumPlotDialog = SpectrumPlotDlg.SpectrumPlotDlg(
                self.waves2D, self.showSpectraPlot
            )
            self.spectrumPlotDialog.show()
        else:
            self.CloseSpectrumPlot()

    def OpenSurfacePlot(self):
        if self.showSurfacePlot.isChecked():
            self.surfacePlotDialog = SurfacePlotDlg.SurfacePlotDlg(
                self.waves2D, self.showSurfacePlot
            )
            self.surfacePlotDialog.show()
            try:
                self.surfacePlotDialog.initSurfaces()
                self.surfacePlotDialog.updatePlots()
            except:
                QtGui.QMessageBox.warning(
                    self,
                    "No 2D wave initialised yet",
                    "Please switch on Two-D Wave in Jonswap Section",
                    QtGui.QMessageBox.Ok,
                )
        else:
            self.CloseSurfacePlot()

    def setupGUI(self):
        # this layout is required to put in the central widget
        self.layout = QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(self.layout)

        # the file menu actions
        fileOpenAction = self.createAction(
            "&Open...",
            self.fileOpen,
            QKeySequence.Open,
            "fileopen",
            "Open an existing image file",
        )
        fileSaveAction = self.createAction(
            "&Save",
            self.fileSave,
            QKeySequence.Save,
            "filesave",
            "Save the image",
        )
        fileSaveAsAction = self.createAction(
            "Save &As...",
            self.fileSaveAs,
            QKeySequence.SaveAs,
            icon="filesaveas",
            tip="Save the image using a new name",
        )
        moveSaveAction = self.createAction(
            "Save &Movie Frames...",
            self.movieSaveAs,
            "Ctrl+M",
            icon="movie",
            tip="Save the movie frame of the 2D surface",
        )
        ExportAction = self.createAction(
            "&Export Spectrum...",
            self.spectrumExportAs,
            "Ctrl+E",
            icon="export",
            tip="Export the complex wave amplitude of the spectrum",
        )
        fileQuitAction = self.createAction(
            "&Quit", self.close, "Ctrl+Q", "filequit", "Close the application"
        )

        # the tool bar actions
        self.playMovieAction = self.createAction(
            text="&Play",
            slot=self.startMovie,
            shortcut="Ctrl+P",
            icon="play",
            tip="Play the movie",
            checkable=True,
            signal="toggled",
            disabled=False
        )

        self.stopMovieAction = self.createAction(
            text="&Stop",
            slot=self.stopMovie,
            shortcut="Ctrl+S",
            icon="stop",
            tip="Stop the movie",
            checkable=True,
            signal="toggled",
        )

        self.pauseMovieAction = self.createAction(
            "P&ause",
            self.pauseMovie,
            "Ctrl+A",
            "pause",
            "Pause the movie",
            True,
            "toggled",
        )

        self.loopMovieAction = self.createAction(
            "&Loop",
            self.loopMovie,
            "Ctrl+L",
            "loop",
            "Play movie in a loop",
            True,
            "toggled",
        )

        self.showSpectraPlot = self.createAction(
            "Spec&tra",
            self.OpenSpectrumPlot,
            "Ctrl+T",
            "plots",
            "Show the spectra",
            True,
            "toggled",
        )

        self.showSurfacePlot = self.createAction(
            "S&urface",
            self.OpenSurfacePlot,
            "Ctrl+U",
            "surface",
            "Show the 2D surface",
            True,
            "toggled",
        )

        # create the file tool bar
        self.fileToolbar = self.addToolBar("File")
        self.fileToolbar.setObjectName("FileToolBar")
        self.addActions(
            self.fileToolbar,
            (fileOpenAction, fileSaveAction, ExportAction, moveSaveAction),
        )

        # create the control tool bar
        self.controlToolBar = self.addToolBar("Control")
        self.controlToolBar.setObjectName("ControlToolBar")
        self.addActions(
            self.controlToolBar,
            [
                self.playMovieAction,
                self.stopMovieAction,
                self.pauseMovieAction,
                self.loopMovieAction,
            ],
        )

        # create the control tool bar
        self.plotToolBar = self.addToolBar("Plots")
        self.plotToolBar.setObjectName("PlotToolBar")
        self.addActions(self.plotToolBar, [self.showSpectraPlot, self.showSurfacePlot])

        # create the file menu
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenuActions = (
            fileOpenAction,
            fileSaveAction,
            fileSaveAsAction,
            ExportAction,
            moveSaveAction,
            fileQuitAction,
        )
        self.fileMenu.aboutToShow.connect(self.updateFileMenu)

        # create the control menu
        self.controlMenu = self.menuBar().addMenu("&Control")
        self.addActions(
            self.controlMenu,
            [
                self.playMovieAction,
                self.stopMovieAction,
                self.pauseMovieAction,
                self.loopMovieAction,
            ],
        )

        # create the plot menu
        self.plotMenu = self.menuBar().addMenu("&Plots")
        self.addActions(self.plotMenu, [self.showSpectraPlot, self.showSurfacePlot])

        # now, set up the centrail widget consisting of a parameter tree at the left and two plots
        # (stacked on top of each other) at the right

        # start with a horizontal splitter dividing the central widget in two parts: left and right
        self.splitter = QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # add the parameter tree to the left side
        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)

        # create a new splitter dividing the top and bottom in two and add that to the right side of
        #  the first splitter
        self.splitter2 = QSplitter()
        self.splitter2.setOrientation(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.splitter2)

        # add the first graph and add it to the top of splitter2
        self.views = []
        self.plots = []
        self.legends = []
        for i in range(2):
            # append the new plot i=0,1 to the lists
            self.views.append(pg.GraphicsLayoutWidget())
            self.plots.append(self.views[-1].addPlot())
            self.legends.append(None)

            # set the plot dependent properties
            if i == 0:
                self.plots[-1].setTitle(
                    "Wave vs distance at time : {:.2f} s".format(0.0)
                )
                self.plots[-1].setXLink("Wave field")
                self.plots[-1].setLabel("bottom", "X-position", units="m")
            else:
                self.plots[-1].setTitle(
                    "Wave vs time at position: {:.2f} s".format(0.0)
                )
                self.plots[-1].setXLink("Point measurement")
                self.plots[-1].setLabel("bottom", "Time", units="m")

            # set the generic properties
            self.plots[-1].setLabel("left", "Amplitude", units="m")
            self.plots[-1].showGrid(x=True, y=False)

            # add the plot to the splitter
            self.splitter2.addWidget(self.views[-1])

        # activate the status bar
        self.sizeLabel = QLabel()
        self.sizeLabel.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.statusbar = self.statusBar()
        self.statusbar.setSizeGripEnabled(False)
        self.statusbar.addPermanentWidget(self.sizeLabel)
        self.statusbar.showMessage("Ready", 5000)

    def sample_at_monitors(self, jswave, i_wave=0):
        """
        sample at the monitor points
        :param jswave:
        :param i_wave:
        :return:
        """

        df = self.wave_monitors[i_wave]

        # create interpolate function
        f_int = interpolate.interp1d(jswave.xpoints, jswave.amplitude)

        k_pos = 0

        for cnt, column in enumerate(df.columns):
            if column == "Time":
                # for the time column only fill in the current time and continue
                self.wave_monitors[i_wave].ix[jswave.t_index, column] = jswave.time
                continue

            # get the velocity of the current monitor point and calculate its current position with
            # the domain
            U_monitor = float(column)
            x_monitor = mod(U_monitor * jswave.time, jswave.Lx) + jswave.xmin

            self.wave_monitor_positions[k_pos] = x_monitor

            # get the amplitude at the current position and store in the data frame
            amplitude = f_int(x_monitor)

            self.wave_monitors[i_wave].ix[jswave.time, column] = amplitude

            k_pos += 1

    def plot_waves_vs_distance(self, plot):
        plot.setTitle(
            "Wave vs distance at time : {:.2f} s".format(self.waves1D[0].time)
        )

        for i, jswave in enumerate(self.waves1D):
            if jswave.plot.show:
                # recalculate the current wave
                jswave.calculate_wave_surface()

                # plot the line
                plot.plot(
                    jswave.xpoints,
                    jswave.amplitude,
                    pen=pg.mkPen(color=jswave.plot.color, width=jswave.plot.linewidth),
                )

                # add some scatter points
                if jswave.plot.scattersize > 0:
                    plot.addItem(
                        pg.ScatterPlotItem(
                            jswave.xpoints,
                            jswave.amplitude,
                            size=jswave.plot.scattersize,
                            pen=jswave.plot.color,
                        )
                    )

                # get the data from the last time step
                if self.monitor_waves:
                    try:
                        time = self.waves2D[0].wave1D.time
                        plot_monitor = "3.0"
                        i_mon = where(
                            self.wave_monitors[0].columns.values == plot_monitor
                        )[0][0]
                        a_mon = self.wave_monitors[0].ix[time, plot_monitor]
                        x_mon = self.wave_monitor_positions[i_mon]
                        point = array([x_mon, a_mon]).reshape((2, 1))
                        self.logger.debug("found {} {} {}".format(i_mon, x_mon, a_mon))
                        self.logger.debug("here {} {} ".format(point, point.shape))
                    except ValueError:
                        self.logger.warning("could not get t_index")
                    else:
                        plot.addItem(
                            pg.ScatterPlotItem(
                                point[0], point[1], pen="r", brush="r", size=5
                            )
                        )

    def updatePlots(self):
        self.clear_plots_and_legends()

        # sample only the first wave
        if self.monitor_waves:
            self.sample_at_monitors(jswave=self.waves1D[0])

        self.plot_waves_vs_distance(self.plots[0])

        for cnt, plot in enumerate(self.plots):
            wave = self.waves2D[cnt].wave1D
            if wave.plot.save_image and self.moviefilebase:
                # dump the picture to a file
                movieframename = self.moviefilebase + "_wave{}_{:06d}.png".format(
                    1, wave.t_index
                )

                self.logger.debug("saving to {}".format(movieframename))
                try:
                    exporter = pg.exporters.ImageExporter(plot)
                    exporter.export(movieframename)
                except:
                    self.logger.warning(
                        "exporting movie frame failed: {}".format(movieframename)
                    )

        if self.waves2D[0].wave1D.twoD and self.surfacePlotDialog:
            if self.moviefilebase is None and self.filename is not None:
                # set movie output file equal to filename if it was not specified
                self.moviefilebase, ext = splitext(self.filename)

            self.surfacePlotDialog.updatePlots(self.moviefilebase)

    def clear_plots_and_legends(self):
        # clear the plots
        for plot in self.plots:
            plot.clear()

        # remove the legends
        for legend in self.legends:
            if legend:
                legend.items = []
                while legend.layout.count() > 0:
                    legend.layout.removeAt(0)

    def addActions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def createAction(self,
                     text=None,
                     slot=None,
                     shortcut=None,
                     icon=None,
                     tip=None,
                     checkable=False,
                     signal="triggered",
                     disabled=False):
        """
        Create a connection between a slot and a signal

        Parameters
        ----------
        text: str, optional
            Text to display in the  menu
        slot:  object, optional
            Slot to connect to the button
        shortcut: object, optional, str
            Shortcut key sequence
        icon: object, optional
            image to show
        tip: str, optional
            String to pop up when hoovering above the button
        checkable: bool, optional
            If true this button can be kept checked
        signal: object, optional
            signal to connect to the slot
        disabled: bool, optional
            If true, gray out button to disable it

        Returns
        -------

        object:
            action

        """

        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/{}.png".format(icon)))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            getattr(action, signal).connect(slot)
        if checkable:
            action.setCheckable(True)
        if disabled:
            action.setEnabled(False)
        return action

    def closeEvent(self, event):
        if self.okToContinue():
            # make sure the movie stops playing
            self.playing = False

            # save the current state of the window and files
            settings = QtCore.QSettings()

            settings.setValue("LastFile", self.filename)

            settings.setValue("RecentFiles", self.recentFiles or [])
            settings.setValue("WaveSimulatorMainWindow/Geometry", self.saveGeometry())
            settings.setValue("WaveSimulatorMainWindow/State", self.saveState())

            # make sure that the dialog is also closed
            self.CloseSpectrumPlot()
            self.CloseSurfacePlot()
        else:
            event.ignore()

    def okToContinue(self):
        if self.dirty:
            reply = QtGui.QMessageBox.question(
                self,
                "Settings changed",
                "Save unsaved changes?",
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel,
            )
            if reply == QtGui.QMessageBox.Cancel:
                return False
            elif reply == QtGui.QMessageBox.Yes:
                self.fileSave()
        return True

    def loadInitialFile(self):
        settings = QtCore.QSettings()

        fname = settings.value("LastFile")

        if fname and QtCore.QFile.exists(fname):
            self.logger.info("loading {}".format(fname))
            self.statusBar().showMessage(
                "Initialising configuration : {} ...".format(basename(fname))
            )
            self.loadFile(fname)
        else:
            self.logger.info("setting initial parameters ")
            self.statusBar().showMessage("Initialising default configuration...")
            self.transfer_parameters()

        self.statusBar().showMessage("Ready.", 5000)

    def updateFileMenu(self):
        self.fileMenu.clear()
        self.addActions(self.fileMenu, self.fileMenuActions[:-1])
        current = self.filename
        recentFiles = []
        for fname in self.recentFiles:
            if fname != current and QtCore.QFile.exists(fname):
                recentFiles.append(fname)
        if recentFiles:
            self.fileMenu.addSeparator()
            for i, fname in enumerate(recentFiles):
                action = QtGui.QAction(
                    QtCore.Icon(":/icon.png"),
                    "&{} {}".format(i + 1, QtCore.QFileInfo(fname).fileName()),
                    self,
                )
                action.setData(fname)
                self.connect(action, QtCore.SIGNAL("triggered()"), self.loadFile)
                self.fileMenu.addAction(action)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.fileMenuActions[-1])

    def fileOpen(self):
        if not self.okToContinue():
            return
        dir = dirname(self.filename) if self.filename is not None else "."

        formats = ["*.{}".format(extension.lower()) for extension in ["cfg"]]

        fname = QtGui.QFileDialog.getOpenFileName(
            self,
            "Wave Simulator  - Choose Configuration",
            dir,
            "Configuration files ({})".format(" ".join(formats)),
        )

        if fname:
            self.loadFile(fname)

    def loadFile(self, fname=None):
        if fname is None:
            action = self.sender()
            if isinstance(action, QtGui.QAction):
                fname = action.data().toString()
                if not self.okToContinue():
                    return
            else:
                return
        if fname:
            self.filename = None
            if self.pars.loadConfiguration(fname):
                self.addRecentFile(fname)
                self.filename = fname
                self.transfer_parameters()
                self.dirty = False

                message = "Loaded {}".format(basename(fname))
                self.logger.debug("{}".format(message))
            else:
                message = "Failed to load the statef file"
                self.logger.debug("{}".format(message))

            self.updateStatus(message)

    def addRecentFile(self, fname):
        if fname is None:
            return
        if fname not in self.recentFiles:
            self.recentFiles = [fname] + self.recentFiles[:8]

    def fileSave(self):
        if self.filename is None:
            self.fileSaveAs()
        else:
            if self.pars.saveConfiguration(self.filename):
                self.updateStatus("Saved as %s" % self.filename)
                self.dirty = False
            else:
                self.updateStatus("Failed to save %s" % self.filename)

    def fileSaveAs(self):
        fname = self.filename if self.filename is not None else "."
        formats = ["*.{}".format(format.lower()) for format in ["cfg"]]
        fname = QtGui.QFileDialog.getSaveFileName(
            self,
            "Wave Simulator - Save Configuration",
            fname,
            "Configuration files ({})".format(" ".join(formats)),
        )
        if fname:
            if "." not in fname:
                fname += ".cfg"
            self.addRecentFile(fname)
            self.filename = fname
            self.fileSave()

    def spectrumExport(self):
        if self.exportfile is None:
            self.spectrumExportAs()
        else:
            # TODO : implement export functions
            if self.waves2D[0].wave1D.twoD:
                # the twoD flag is checked: export the 2D wave numbers
                for i, wave in enumerate(self.waves2D):
                    if wave.wave1D.plot.show:
                        fname = sub(
                            "[_1|_2]*.w2d",
                            "_{}.w2d".format(i + 1, ".w2d"),
                            self.exportfile,
                        )
                        self.updateStatus("Export 2D wave nodes to {}".format(fname))
                        wave.export_complex_amplitudes(fname)
                    else:
                        self.logger.info(
                            "wave {} not active, not exporting".format(i + 1)
                        )
            else:
                # the twoD flag is not checked, export the 1D
                for i, wave in enumerate(self.waves1D):
                    if wave.plot.show:
                        fname = sub(
                            "[_1|_2]*.w1d",
                            "_{}.w1d".format(i + 1, ".w1d"),
                            self.exportfile,
                        )
                        self.updateStatus("Export 1D wave nodes to {}".format(fname))
                        try:
                            wave.export_complex_amplitudes(fname)
                        except AttributeError as err:
                            self.logger.warning(err)

                        if i == 0:
                            base, ext = splitext(fname)
                            fname = base + "_mon.xls"
                            self.updateStatus(
                                "Export 1D wave monitor to {}".format(fname)
                            )
                            if self.monitor_waves:
                                with pd.ExcelWriter(fname, append=False) as writer:
                                    self.wave_monitors[i].to_excel(writer)
                    else:
                        self.logger.info(
                            "wave {} not active, not exporting".format(i + 1)
                        )

    def spectrumExportAs(self):
        fname = self.exportfile if self.exportfile is not None else "."
        formats = ["*.{}".format(format.lower()) for format in ["h5", "cfg", "xls"]]
        fname = QtGui.QFileDialog.getSaveFileName(
            self,
            "Wave Simulator - Export Wave Aplitudes",
            fname,
            "Wave amplitude files ({})".format(" ".join(formats)),
        )
        if fname:
            if "." not in fname:
                if self.waves2D[0].wave1D.twoD:
                    fname += "_2D.h5"
                else:
                    fname += "_1D.h5"
            self.exportfile = fname
            self.spectrumExport()

    def movieSaveAs(self):
        fname = self.moviefilebase if self.moviefilebase is not None else "."
        formats = ["*.{}".format(format.lower()) for format in ["png", "h5", "cfg"]]
        fname = QtGui.QFileDialog.getSaveFileName(
            self,
            "Wave Simulator - Save Movie Frames",
            fname,
            "Image files ({})".format(" ".join(formats)),
        )
        if fname:
            fname = sub("[_\d*]*.png", "", fname)

            self.moviefilebase = fname

    def updateStatus(self, message):
        self.statusBar().showMessage(message, 5000)
        if self.filename is not None:
            self.setWindowTitle("Wave Simulator - %s[*]" % basename(self.filename))
        else:
            self.setWindowTitle("Wave Simulator[*]")
        self.setWindowModified(self.dirty)

    def addRecentFile(self, fname):
        if fname is None:
            return

        if fname not in self.recentFiles:
            self.recentFiles = [fname] + self.recentFiles[:8]

    def updateFileMenu(self):
        self.fileMenu.clear()
        self.addActions(self.fileMenu, self.fileMenuActions[:-1])
        current = self.filename
        recentFiles = []
        for fname in self.recentFiles:
            if fname != current and QtCore.QFile.exists(fname):
                recentFiles.append(fname)
        if recentFiles:
            self.fileMenu.addSeparator()
            for i, fname in enumerate(recentFiles):
                action = QtGui.QAction(
                    QIcon(":/icon.png"),
                    "&{} {}".format(i + 1, QtCore.QFileInfo(fname).fileName()),
                    self,
                )
                action.setData(fname)
                action.triggered.connect(self.loadFile())
                self.fileMenu.addAction(action)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.fileMenuActions[-1])


def main():
    # Create a GL View widget to display data

    app = pg.mkQApp()
    app.setOrganizationName("HMC Heerema Marine Contractors")
    app.setOrganizationDomain("hmc-heerema.com")
    app.setApplicationName("Wave Simulator")
    app.setWindowIcon(QIcon(":icon.png"))
    win = WaveSimulatorGUI()
    win.setWindowTitle("Wave Simulator")
    win.resize(1100, 700)
    win.show()

    app.exec_()


def parse_the_command_line_arguments(__version__):
    """

    Parameters
    ----------
    __version__ :


    Returns
    -------

    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # parse the command line to set some options2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser = argparse.ArgumentParser(
        description="Sequence tool for waiting on weather analyzes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # set the verbosity level command line arguments
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Be quiet: no output",
        action="store_const",
        dest="log_level",
        const=logging.WARNING,
    )
    parser.add_argument(
        "--write_log_to_file",
        action="store_true",
        help="Write the logging information to file",
    )
    parser.add_argument(
        "--log_file_base", default="log", help="Default name of the logging output"
    )
    parser.add_argument(
        "--log_file_debug",
        help="Print lots of debugging statements to file",
        action="store_const",
        dest="log_level_file",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "--log_file_verbose",
        help="Be verbose to file",
        action="store_const",
        dest="log_level_file",
        const=logging.INFO,
    )
    parser.add_argument(
        "--log_file_quiet",
        help="Be quiet: no output to file",
        action="store_const",
        dest="log_level_file",
        const=logging.WARNING,
    )
    # parse the command line
    args = parser.parse_args()

    return args, parser


if __name__ == "__main__":
    sys.argv = clear_argument_list(sys.argv)

    args, parser = parse_the_command_line_arguments(__version__)

    if args.write_log_to_file:
        # http://stackoverflow.com/questions/29087297/
        # is-there-a-way-to-change-the-filemode-for-a-logger-object-that-is-not-configured
        log_file_base = args.log_file_base
        sys.stderr = open(log_file_base + ".err", "w")
    else:
        log_file_base = None

    logger = create_logger(
        file_log_level=args.log_level_file,
        console_log_level=args.log_level,
        log_file=log_file_base,
    )

    main()
