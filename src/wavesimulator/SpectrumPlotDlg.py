import logging
import sys

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore


class SpectrumPlotDlg(QtGui.QDialog):
    def __init__(self, waves, showSpectraPlot, parent=None):
        super(SpectrumPlotDlg, self).__init__(parent)

        self.waves = waves

        self.showSpectraPlot = showSpectraPlot

        # initialse the logger
        self.logger = logging.getLogger(__name__)

        # make sure the dialog is close (not hidden)
        # actually, I want to hide it
        # self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)

        self.radiobuttons = []
        self.radiobuttons.append(QtGui.QRadioButton())
        self.radiobuttons[-1].setText("Spectrum K/Omega")
        self.radiobuttons[-1].setChecked(True)
        self.radiobuttons.append(QtGui.QRadioButton())
        self.radiobuttons[-1].setText("Spectrum K/Mod")
        self.radiobuttons.append(QtGui.QRadioButton())
        self.radiobuttons[-1].setText("Spectrum K/Phase")
        self.radiobuttons.append(QtGui.QRadioButton())
        self.radiobuttons[-1].setText("Spectrum K Module/Spreading function")
        self.radiobuttons.append(QtGui.QRadioButton())
        self.radiobuttons[-1].setText("k-theta Amplitude/k-theta Phase")
        self.radiobuttons.append(QtGui.QRadioButton())
        self.radiobuttons[-1].setText("kx/ky Amplitude/kx-ky Phase")

        self.checkBox_wavelines = []
        self.checkBox_wavepoints = []
        self.checkBox_triangles = []
        for i in range(2):
            self.checkBox_wavelines.append(QtGui.QCheckBox())
            self.checkBox_wavelines[-1].setText("Plot Wave {}".format(i + 1))
            self.checkBox_wavelines[-1].setChecked(True)
            self.connect(self.checkBox_wavelines[-1], QtCore.SIGNAL("toggled(bool)"),
                         self.checkToggled)
            self.checkBox_wavepoints.append(QtGui.QCheckBox())
            self.checkBox_wavepoints[-1].setText("Add Scatter Points Wave {}".format(i + 1))
            self.checkBox_wavepoints[-1].setChecked(True)
            self.connect(self.checkBox_wavepoints[-1], QtCore.SIGNAL("toggled(bool)"),
                         self.checkToggled)
            self.checkBox_triangles.append(QtGui.QCheckBox())
            self.checkBox_triangles[-1].setText("Add Triangles at key points {}".format(i + 1))
            self.checkBox_triangles[-1].setChecked(True)
            self.connect(self.checkBox_triangles[-1], QtCore.SIGNAL("toggled(bool)"),
                         self.checkToggled)

        # stack = QtGui.QStackedWidget()

        self.view = pg.GraphicsLayoutWidget()

        # create two plots above each other. Depending on the radio button, the two plots will display
        # the spectrum modules vs k (top) and omega (bottom) or the spectrum modules vs (k) and phase vs k (bottom
        self.plots = []
        self.legends = []
        # this will only initialise the plot axis
        for i in range(2):
            self.plots.append(self.view.addPlot())
            self.legends.append(None)

            # link to current plot
            plot = self.plots[-1]

            # generic properties
            plot.showGrid(x=True, y=False)

            # plot dependent properties
            if i == 0:
                plot.setXLink('Modulus')
                plot.setLabel('left', "Modulus", units='m^2 m')
                plot.setLabel('bottom', "k", units='1/m')
                # go the the next plot
                self.view.nextRow()
            else:
                plot.setXLink("Phase")
                plot.setLabel('left', "Phase", units='rad')
                plot.setLabel('bottom', "k", units='1/m')

        # now plot the lines
        self.updatePlots()

        # create the plot screen and the buttons below it
        grid = QtGui.QGridLayout()
        grid.addWidget(self.view, 0, 0, 1, 5)

        n = 0
        for j in range(2):
            for i in range(3):
                grid.addWidget(self.radiobuttons[n], 1 + i, j)
                n += 1

        for i, cb in enumerate(self.checkBox_wavelines):
            grid.addWidget(self.checkBox_wavelines[i], 1, 2 + i)
        for i, cb in enumerate(self.checkBox_wavepoints):
            grid.addWidget(self.checkBox_wavepoints[i], 2, 2 + i)
        for i, cb in enumerate(self.checkBox_triangles):
            grid.addWidget(self.checkBox_triangles[i], 3, 2 + i)

        grid.addWidget(self.buttonBox, 2, 4)

        self.setLayout(grid)

        # connect the buttons to the apply and close slots

        self.connect(self.buttonBox, QtCore.SIGNAL("rejected()"),
                     self, QtCore.SLOT("close()"))

        for radiobutton in self.radiobuttons:
            self.connect(radiobutton, QtCore.SIGNAL("toggled(bool)"),
                         self.radioToggled)

        # set the dialog position and size based on the last open session
        settings = QtCore.QSettings()


        self.restoreGeometry(settings.value("SpectrumPlotDlg/Geometry",
                                            QtCore.QByteArray()))

        self.setWindowTitle("Spectrum Plots")

    def checkToggled(self, flag):
        self.updatePlots()

    def radioToggled(self, set):

        if set:
            self.updatePlots()

    def triangle_with_text(self, curve, labeltext, pos=None, index=0, triangleAngle=-90, anchorPos=(0.5, 0, 0)):
        """
        this routine places a triangle at a given location of a plot curve with a label above it

        :param curve: the link to the current plot line
        :param labeltext: a text to be displayed
        :param pos: the location to plot the text
        :param triangleAngle: the angle of the triangle pointing to the location
        :param anchorPos: some offset
        :return: the point containing the point
        """
        curvePoint = pg.CurvePoint(curve)

        if pos is not None:
            curvePoint.setPos(pos)
        else:
            curvePoint.setIndex(index)

        text = pg.TextItem(labeltext, anchor=anchorPos)
        text.setParentItem(curvePoint)
        arrow = pg.ArrowItem(angle=triangleAngle)
        arrow.setParentItem(curvePoint)

        return curvePoint

    # @profile
    def plot_spectrum_modulus_vs_k(self, plot):

        # create the plot for the spectrum and the modules

        wave = self.waves[0].wave1D
        kmax = np.max(wave.kx_nodes)
        amax = np.max(wave.spectrumK)

        # set the label positions
        lx = 0.8 * kmax
        ly = [0.9 * amax, 0.4 * amax]

        # add the title and axis labels
        plot.setTitle('Jonswap Spectrum vs wave number')
        plot.setLabel('left', "S(k)", units='m^2 s')
        plot.setLabel('bottom', "k", units='rad/m')

        for i, wave2D in enumerate(self.waves):
            wave = wave2D.wave1D
            color = wave.plot.color

            # skip this line if the checkbox is not checked for both the lines and scatter points
            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            # create the line plot and store in 'curve' in order to use later for adding the triangles
            if self.checkBox_wavelines[i].isChecked():
                curve = plot.plot(wave.kx_nodes, wave.spectrumK, pen=color)

                # only for the first line (i=0) we need to add the triangles and labels
                if self.checkBox_triangles[i].isChecked():
                    # add the triangles and labels to indicate the key point
                    label = "E[k<{:.3f}]/E={:.1f}%".format(wave.k_low, wave.E_limit_low * 100)
                    plot.addItem(self.triangle_with_text(curve, label, wave.k_low / kmax, anchorPos=(1, 2)))

                    label = "E[k<{:.2f}]/E={:.1f}%".format(wave.k_high, wave.E_limit_high * 100)
                    plot.addItem(self.triangle_with_text(curve, label, wave.k_high / kmax, anchorPos=(0, 2)))

                    label = "k_peak={:.2f}/m; lambda_peak={:.1f} m".format(wave.k_peak, 2 * np.pi / wave.k_peak)
                    plot.addItem(self.triangle_with_text(curve, label, wave.k_peak / kmax, anchorPos=(.5, 2)))

            if self.checkBox_wavepoints[i].isChecked():
                # add scatter points to the curve
                plot.addItem(pg.ScatterPlotItem(wave.kx_nodes, wave.spectrumK, size=5, pen=color))

            # add a label with box for the current wave
            sigma = np.sqrt(wave.varianceK)
            k_min = min(np.diff(wave.kx_nodes))
            Lmax = 2 * np.pi / k_min
            txt = pg.TextItem("Variance: {:.2f}\nHs={:.1f}\nN={:d}\ndelta_k_min={:12.4e} rad/m\nLmax={:.0f} m".format(
                sigma ** 2, 4 * sigma, wave.kx_nodes.size, k_min, Lmax), border='b', anchor=(0, 1), color=color)
            txt.setPos(lx, ly[i])
            plot.addItem(txt, pen=color)

    # @profile
    def plot_spectrum_phase_vs_k(self, plot):
        # create the plot for the spectrum and the modules

        plot.setTitle('Amplitude spectrum Phase')
        plot.setLabel('left', "phi(k)", units='rad')
        plot.setLabel('bottom', "k", units='rad/m')

        for i, wave2D in enumerate(self.waves):
            wave = wave2D.wave1D
            color = wave.plot.color

            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            # create the line plot and store in 'curve' in order to use later for adding the triangles
            if self.checkBox_wavelines[i].isChecked():
                plot.plot(wave.kx_nodes, np.angle(wave.complex_Amplitudes), pen=color)

            if self.checkBox_wavepoints[i].isChecked():
                plot.addItem(pg.ScatterPlotItem(wave.kx_nodes, np.angle(wave.complex_Amplitudes), size=5, pen=color))

    # @profile
    def plot_spectrum_mod_vs_k(self, plot):
        # create the plot for the spectrum and the modules

        plot.setTitle('Amplitude spectrum mod Amplitude')
        plot.setLabel('left', "abs(A)(k)", units='m')
        plot.setLabel('bottom', "k", units='rad/m')

        for i, wave2D in enumerate(self.waves):
            wave = wave2D.wave1D
            color = wave.plot.color

            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            # create the line plot and store in 'curve' in order to use later for adding the triangles
            if self.checkBox_wavelines[i].isChecked():
                plot.plot(wave.kx_nodes, abs(wave.complex_Amplitudes), pen=color)

            if self.checkBox_wavepoints[i].isChecked():
                plot.addItem(pg.ScatterPlotItem(wave.kx_nodes, abs(wave.complex_Amplitudes), size=5, pen=color))

    # @profile
    def plot_spectrum_modulus_vs_omega(self, plot):

        # Bottom plot: the spectrum vs omega
        plot.setTitle('Jonswap Spectrum vs Omega')
        plot.setLabel('left', "S(omega)", units='[m^2 s]')
        plot.setLabel('bottom', "omega", units='rad/s')

        wave = self.waves[0].wave1D
        Wmax = np.max(wave.omega_dispersion)
        ymax = np.max(wave.spectrumW)

        # set the label positions
        lx = 0.8 * Wmax
        ly = [0.8 * ymax, 0.5 * ymax]

        for i, wave2D in enumerate(self.waves):
            wave = wave2D.wave1D
            color = wave.plot.color

            # loop over the waves. The top and bottom can both contain information of each wave
            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            if self.checkBox_wavelines[i].isChecked():
                curve = plot.plot(wave.omega_dispersion, wave.spectrumW, pen=color, width=5)
                if self.checkBox_triangles[i].isChecked():
                    # for the first plot only, add the triangles if the curve is plot
                    label = "E[W<{:.3f}]/E={:.1f}%".format(wave.W_low, wave.E_limit_low * 100)
                    plot.addItem(self.triangle_with_text(curve, label, index=wave.iW_low, anchorPos=(1, 2)))

                    label = "E[W<{:.2f}]/E={:.1f}%".format(wave.W_high, wave.E_limit_high * 100)
                    plot.addItem(self.triangle_with_text(curve, label, index=wave.iW_high, anchorPos=(0, 2)))

                    label = "W_peak={:.2f}/s; T_peak={:.1f} s".format(wave.W_peak, 2 * np.pi / wave.W_peak)
                    plot.addItem(self.triangle_with_text(curve, label, index=wave.iW_peak, anchorPos=(.5, 2)))

            if self.checkBox_wavepoints[i].isChecked():
                plot.addItem(pg.ScatterPlotItem(wave.omega_dispersion, wave.spectrumW, size=5, pen=color))

            sigma = np.sqrt(wave.varianceK)
            txt = pg.TextItem("Variance: {:.2f}\nHs={:.1f}".format(sigma ** 2, 4 * sigma), border='b', anchor=(0, 1))
            txt.setPos(lx, ly[i])
            plot.addItem(txt)

    # @profile
    def plot_spreading_vs_theta(self, plot):

        # Bottom plot: the spectrum vs omega
        plot.setTitle('Jonswap Spreading Function vs Theta')
        plot.setLabel('left', "PDF(theta)", units='1/rad')
        plot.setLabel('bottom', "theta", units='rad')

        xmax = np.max(self.waves[0].theta_points)
        ymax = np.max(self.waves[0].D_spread)

        # set the label positions
        lx = 0.8 * xmax
        ly = [0.8 * ymax, 0.5 * ymax]

        for i, wave in enumerate(self.waves):
            color = wave.wave1D.plot.color

            # loop over the waves. The top and bottom can both contain information of each wave
            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            if self.checkBox_wavelines[i].isChecked():
                curve = plot.plot(wave.theta_points, wave.D_spread, pen=color, width=5)
                if i == 0:
                    # for the first plot only, add the triangles if the curve is plot
                    label = "Tlow={:.3f}".format(wave.theta_low)
                    plot.addItem(self.triangle_with_text(curve, label, float(wave.theta_low / xmax), anchorPos=(1, 2)))

                    label = "Thigh={:.2f}".format(wave.theta_high)
                    plot.addItem(self.triangle_with_text(curve, label, float(wave.theta_high / xmax), anchorPos=(0, 2)))

                    label = "Theta_peak={:.2f} rad".format(wave.theta_peak)
                    plot.addItem(
                        self.triangle_with_text(curve, label, float(wave.theta_peak / xmax), anchorPos=(.5, 2)))

            if self.checkBox_wavepoints[i].isChecked():
                plot.addItem(pg.ScatterPlotItem(wave.theta_points, wave.D_spread, size=5, pen=color))

            area = np.sum(wave.D_spread) * np.diff(wave.theta_points)[0]
            txt = pg.TextItem("Area: {:.2f}\nN={:d}".format(area, wave.theta_points.size),
                              border='b', anchor=(0, 1), color=color)
            txt.setPos(lx, ly[i])
            plot.addItem(txt)

    # @profile
    def plot_k_theta_spectrum_modulus(self, plot):

        plot.setTitle('Jonswap k-theta scatter plot of modulus')
        plot.setLabel('left', "index theta", units='-')
        plot.setLabel('bottom', "index k", units='-')

        xmax = np.max(self.waves[0].wave1D.kx_nodes)
        ymax = np.max(self.waves[0].theta_points)

        # set the label positions
        lx = 0.8 * xmax
        ly = [0.8 * ymax, 0.5 * ymax]

        for i, wave in enumerate(self.waves):
            color = wave.wave1D.plot.color

            # loop over the waves. The top and bottom can both contain information of each wave
            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            if self.checkBox_wavepoints[i].isChecked():
                img = pg.ImageItem(abs(wave.E_wave_density_complex_amplitudes), color=color)
                plot.addItem(img)

    # @profile
    def plot_k_theta_spectrum_phase(self, plot):

        plot.setTitle('Jonswap k-theta scatter plot of phase')
        plot.setLabel('left', "index theta", units='-')
        plot.setLabel('bottom', "index k", units='-')

        xmax = np.max(self.waves[0].wave1D.kx_nodes)
        ymax = np.max(self.waves[0].theta_points)

        # set the label positions
        lx = 0.8 * xmax
        ly = [0.8 * ymax, 0.5 * ymax]

        for i, wave in enumerate(self.waves):
            color = wave.wave1D.plot.color

            # loop over the waves. The top and bottom can both contain information of each wave
            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            if self.checkBox_wavepoints[i].isChecked():
                img = pg.ImageItem(np.angle(wave.E_wave_density_complex_amplitudes), color=color)
                plot.addItem(img)

    # @profile
    def plot_kx_ky_spectrum_modulus(self, plot):
        plot.setTitle('Jonswap kx-ky scatter plot of modulus')
        plot.setLabel('left', "index kx", units='-')
        plot.setLabel('bottom', "index ky", units='-')

        xmax = np.max(self.waves[0].wave1D.kx_nodes)
        ymax = np.max(self.waves[0].theta_points)

        # set the label positions
        lx = 0.8 * xmax
        ly = [0.8 * ymax, 0.5 * ymax]

        for i, wave in enumerate(self.waves):
            color = wave.wave1D.plot.color

            # loop over the waves. The top and bottom can both contain information of each wave
            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            if self.checkBox_wavepoints[i].isChecked():
                img = pg.ImageItem(abs(wave.E_wave_density_complex_amplitudes), color=color)
                plot.addItem(img)

    # @profile
    def plot_kx_ky_spectrum_phase(self, plot):
        plot.setTitle('Jonswap kx-ky scatter plot of phase')
        plot.setLabel('left', "index ky", units='-')
        plot.setLabel('bottom', "index kx", units='-')

        xmax = np.max(self.waves[0].wave1D.kx_nodes)
        ymax = np.max(self.waves[0].theta_points)

        # set the label positions
        lx = 0.8 * xmax
        ly = [0.8 * ymax, 0.5 * ymax]

        for i, wave in enumerate(self.waves):
            color = wave.wave1D.plot.color

            # loop over the waves. The top and bottom can both contain information of each wave
            if not (self.checkBox_wavelines[i].isChecked() or self.checkBox_wavepoints[i].isChecked()):
                continue

            if self.checkBox_wavepoints[i].isChecked():
                img = pg.ImageItem(np.angle(wave.E_wave_density_complex_amplitudes), color=color)
                plot.addItem(img)

    # @profile
    def updatePlots(self):

        # clear the previous plot and fill with the new plots depending on the radio button

        self.clear_plots_and_legends()

        if self.radiobuttons[0].isChecked():

            # create the top plot with the spectrum vs k
            self.plot_spectrum_modulus_vs_k(self.plots[0])

            # create the bottom plot with the spectrum vs omega
            self.plot_spectrum_modulus_vs_omega(self.plots[1])

        elif self.radiobuttons[1].isChecked():
            # plot spectra for wave number modulus and phase

            # create the top plot with the spectrum vs k
            self.plot_spectrum_modulus_vs_k(self.plots[0])

            self.plot_spectrum_mod_vs_k(self.plots[1])

        elif self.radiobuttons[2].isChecked():
            # plot spectra for wave number modulus and phase

            # create the top plot with the spectrum vs k
            self.plot_spectrum_modulus_vs_k(self.plots[0])

            self.plot_spectrum_phase_vs_k(self.plots[1])
        elif self.radiobuttons[3].isChecked():

            self.plot_spectrum_modulus_vs_k(self.plots[0])

            self.plot_spreading_vs_theta(self.plots[1])
        elif self.radiobuttons[4].isChecked():
            self.plot_k_theta_spectrum_modulus(self.plots[0])
            self.plot_k_theta_spectrum_phase(self.plots[1])
        elif self.radiobuttons[5].isChecked():
            self.plot_kx_ky_spectrum_modulus(self.plots[0])
            self.plot_kx_ky_spectrum_phase(self.plots[1])
        else:
            self.logger.warning("radio button not defined")

    def closeEvent(self, event):

        # before closing the window store its size and position
        settings = QtCore.QSettings()
        settings.setValue("SpectrumPlotDlg/Geometry", self.saveGeometry())

        # uncheck the showSpectraPlot button before closing the dialog
        if self.showSpectraPlot.isChecked():
            self.showSpectraPlot.setChecked(False)

    # @profile
    def clear_plots_and_legends(self):

        # clear the plots and manually remove the legend keys

        for plot in self.plots:
            plot.clear()

        # clear the legend keys
        for legend in self.legends:
            if legend:
                legend.items = []
                while legend.layout.count() > 0:
                    legend.layout.removeAt(0)
