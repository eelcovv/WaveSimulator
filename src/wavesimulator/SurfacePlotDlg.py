import sys
import logging

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from numpy import std
import pyqtgraph.opengl as gl
import pyqtgraph.ptime as ptime
import h5py

from os.path import splitext



class SurfacePlotDlg(QtGui.QDialog):
    def __init__(self, waves, showSurfacePlot,parent=None):
        super(SurfacePlotDlg, self).__init__(parent)

        self.waves = waves

        self.showSurfacePlot = showSurfacePlot

        # initialse the logger
        self.logger = logging.getLogger(__name__)

        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)

        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('GLSurfacePlot of the 2D wave field')
        self.view.setCameraPosition(distance=500)

        self.image_base = "surface"
        self.checkBox_saveToFile = QtGui.QCheckBox()
        self.checkBox_saveToFile.setText("Save frames to File {}".format(self.image_base))
        self.checkBox_saveToFile.setChecked(False)
        self.connect(self.checkBox_saveToFile, QtCore.SIGNAL("toggled(bool)"),
                         self.checkToggled)

        # initialise list of surface.
        self.surfaces = []
        self.gridlines = None

        # create the plot screen and the buttons below it
        grid = QtGui.QGridLayout()
        grid.addWidget(self.view, 0, 0, 1, 5)

        grid.addWidget(self.buttonBox, 2, 4)

        self.setLayout(grid)

        # connect the buttons to the apply and close slots

        self.connect(self.buttonBox, QtCore.SIGNAL("rejected()"),
                     self, QtCore.SLOT("close()"))

        # set the dialog position and size based on the last open session
        settings = QtCore.QSettings()

        self.restoreGeometry(settings.value("SurfacePlotDlg/Geometry",
                                            QtCore.QByteArray()))


    def checkToggled(self, flag):
        self.updatePlots()

    def radioToggled(self, set):

        if set:
            self.updatePlots()

    # @profile
    def initSurfaces(self):
        ## create a surface plot, tell it to use the 'heightColor' shader
        ## since this does not require normal vectors to render (thus we
        ## can set computeNormals=False to save time when the mesh updates)
        for surface in self.surfaces:
            if surface:
                self.view.removeItem(surface)

        if self.gridlines:
            self.view.removeItem(self.gridlines)

        self.scalefactor=10
        for wave in self.waves:
            self.scalefactor=max(wave.xmax,wave.ymax)/10


        self.setWindowTitle("Grid scale {:.0f} m".format(self.scalefactor))


        # Add a grid to the view
        self.gridlines = gl.GLGridItem()
        self.gridlines.scale(self.scalefactor, self.scalefactor, 1)
        self.gridlines.setDepthValue(10)  # draw grid after surfaces since they may be translucent
        self.gridlines.translate(self.waves[0].xmid, self.waves[0].ymid, 0)
        self.view.addItem(self.gridlines)

        self.surfaces = []
        for i, wave in enumerate(self.waves):
            qcol = wave.wave1D.plot.color
            [R, G, B, T] = qcol.getRgbF()
            if wave.wave1D.plot.show:
                self.surfaces.append(gl.GLSurfacePlotItem(x=wave.xpoints, y=wave.ypoints,
                                                          shader='shaded', color=qcol.getRgbF()))

                self.view.addItem(self.surfaces[-1])
            else:
                self.surfaces.append(None)

    # @profile
    def updatePlots(self,moviefilebase=None):

        # clear the previous plot and fill with the new plots depending on the radio button
        stime = ptime.time()
        hstring=""
        for i, surf in enumerate(self.surfaces):

            wave = self.waves[i]

            if wave.wave1D.plot.show:
                wave.calculate_wave_surface()
                hstring+=" /Hs({})={:10.1f} m".format(i,std(wave.amplitude)*4)
                if surf:
                    surf.setData(z=wave.amplitude)
                    if wave.wave1D.plot.save_image and moviefilebase:
                        # dump the picture to a file
                        movieframename = moviefilebase+"_wave{}_{:06d}.png".format(i+1,wave.wave1D.t_index)
                        self.logger.debug("saving to {}".format(movieframename))
                        try:
                            self.view.grabFrameBuffer().save(movieframename)
                        except:
                            self.logger.warning("exporting movie frame failed: {}".format(movieframename))
                    if wave.wave1D.plot.save_data and moviefilebase:
                        # dump the data to a file
                        datafilename = moviefilebase+"_wave{}_{:06d}.h5".format(i+1,wave.wave1D.t_index)
                        self.logger.debug("saving to {}".format(datafilename))
                        if wave.wave1D.t_index == 0 :
                            # for the first time only, also dump the xy mesh into a separate file
                            with h5py.File(moviefilebase+"_wave{}_xy-mesh".format(i+1), "w") as hf:
                                hf.create_dataset('X', data=wave.xy_mesh[0])
                                hf.create_dataset('Y', data=wave.xy_mesh[1])
                        # for all the other frames, dump the amplitude
                        with h5py.File(datafilename, "w") as hf:
                            hf.create_dataset('amplitude', data=wave.amplitude)
            else:
                hstring+=" /Hs({})=N.A."

        self.setWindowTitle("Grid scale {:.0f} m, {} (calculated in {:10.4f} s)".format(
                    self.scalefactor,hstring,ptime.time() - stime))


    def closeEvent(self, event):

        # before closing the window store its size and position
        settings = QtCore.QSettings()
        settings.setValue("SurfacePlotDlg/Geometry", self.saveGeometry())

        # uncheck the showSurfacePlot button before closing the dialog
        if self.showSurfacePlot.isChecked():
            self.showSurfacePlot.setChecked(False)
