__author__ = 'eelcovv'

import logging

import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from numpy import pi
from pyqtgraph.Qt import QtCore
from pyqtgraph.configfile import ParseError
from pyqtgraph.parametertree import Parameter


class DomainParameters(pTypes.GroupParameter):
    # create a group of parameters of the time span and resolution which are all dependending on
    # each other so that they should mutually change
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True

        self.logger = logging.getLogger(__name__)

        pTypes.GroupParameter.__init__(self, **opts)

        x_min = 0.0
        x_max = 1000.0
        lx = x_max - x_min
        nx = 128
        dx = lx / (nx - 1)

        lk_max = lx
        kx_min = 0
        kx_max = 2 * pi / dx
        nkx = 1000
        dkx = (kx_max - kx_min) / nkx
        kxn = kx_max / 2

        # the members of the group as they appear in the menu
        self.addChild(dict(name="X-axis Mesh", type="group", children=[
            dict(name='X minimum', type='float', value=x_min, step=1, siPrefix=True, suffix='m'),
            dict(name='X maximum', type='float', value=x_max, step=100, siPrefix=True, suffix='m'),
            dict(name='Length X', type='float', value=lx, step=100, limits=(0.01, None),
                 siPrefix=True, suffix='m'),
            dict(name='Number of x nodes', type='int', value=nx, limits=(2, None)),
            dict(name='Delta x', type='float', value=dx, siPrefix=True, suffix='m')
        ]
                           ))

        # the wave vector parameters
        self.addChild(dict(name="KX-axis Wave vectors", type="group", children=[
            dict(name='Minimum kx', type='float', value=kx_min, suffix=' rad/m'),
            dict(name='Maximum kx', type='float', value=kx_max, limits=(0.001, None),
                 suffix=' rad/m'),
            dict(name='Number of kx nodes', type='int', value=nkx, limits=(3, None)),
            dict(name='Delta kx', type='float', value=dkx, suffix=' rad/m'),
            dict(name='Nyquist kx', type='float', value=kxn, suffix=' rad/m', readonly=True),
            dict(name='Maximum wave length x', type='float', value=lk_max, suffix='m',
                 readonly=True)
        ]
                           ))

        self.addChild(dict(name="Y-axis Mesh", type="group", children=[
            dict(name='Y minimum', type='float', value=kx_min, step=1, siPrefix=True, suffix='m'),
            dict(name='Y maximum', type='float', value=kx_max, step=100, siPrefix=True, suffix='m'),
            dict(name='Length Y', type='float', value=lx, step=100, limits=(0.01, None),
                 siPrefix=True, suffix='m'),
            dict(name='Number of y nodes', type='int', value=nx, limits=(2, None)),
            dict(name='Delta y', type='float', value=dx, siPrefix=True, suffix='m')
        ]
                           ))

        self.addChild(dict(name="KY-axis Wave vectors", type="group", children=[
            dict(name='Minimum ky', type='float', value=kx_min, suffix=' rad/m'),
            dict(name='Maximum ky', type='float', value=kx_max, limits=(0.001, None),
                 suffix=' rad/m'),
            dict(name='Number of ky nodes', type='int', value=nkx, limits=(3, None)),
            dict(name='Delta ky', type='float', value=dkx, suffix=' rad/m'),
            dict(name='Nyquist ky', type='float', value=kxn, suffix=' rad/m', readonly=True),
            dict(name='Maximum wave length y', type='float', value=lk_max, suffix='m',
                 readonly=True)
        ]
                           ))

        self.addChild(dict(name="Polar mesh settings", type="group", children=[
            dict(name='Number of theta nodes', type='int', value=60, limits=(1, None))
        ]
                           ))

        self.kx_min = self.param("KX-axis Wave vectors").names['Minimum kx']
        self.kx_max = self.param("KX-axis Wave vectors").names['Maximum kx']
        self.n_kx_nodes = self.param("KX-axis Wave vectors").names['Number of kx nodes']
        self.delta_kx = self.param("KX-axis Wave vectors").names['Delta kx']
        self.kx_nyquist = self.param("KX-axis Wave vectors").names["Nyquist kx"]
        self.max_lambda_x = self.param("KX-axis Wave vectors").names["Maximum wave length x"]

        self.n_theta_nodes = self.param("Polar mesh settings").names['Number of theta nodes']

        self.kx_min.sigValueChanged.connect(self.kx_range_Changed)
        self.kx_max.sigValueChanged.connect(self.kx_range_Changed)
        self.n_kx_nodes.sigValueChanged.connect(self.n_kx_nodes_Changed)
        self.delta_kx.sigValueChanged.connect(self.delta_kx_Changed)

        self.ky_min = self.param("KY-axis Wave vectors").names['Minimum ky']
        self.ky_max = self.param("KY-axis Wave vectors").names['Maximum ky']
        self.n_ky_nodes = self.param("KY-axis Wave vectors").names['Number of ky nodes']
        self.delta_ky = self.param("KY-axis Wave vectors").names['Delta ky']
        self.ky_nyquist = self.param("KY-axis Wave vectors").names["Nyquist ky"]
        self.max_lambda_y = self.param("KY-axis Wave vectors").names["Maximum wave length y"]

        self.ky_min.sigValueChanged.connect(self.ky_range_Changed)
        self.ky_max.sigValueChanged.connect(self.ky_range_Changed)
        self.n_ky_nodes.sigValueChanged.connect(self.n_ky_nodes_Changed)
        self.delta_ky.sigValueChanged.connect(self.delta_ky_Changed)

        self.xmin = self.param("X-axis Mesh").names["X minimum"]
        self.xmax = self.param("X-axis Mesh").names['X maximum']
        self.Lx = self.param("X-axis Mesh").names['Length X']
        self.nx = self.param("X-axis Mesh").names['Number of x nodes']
        self.delta_x = self.param("X-axis Mesh").names['Delta x']

        self.ymin = self.param("Y-axis Mesh").names['Y minimum']
        self.ymax = self.param("Y-axis Mesh").names['Y maximum']
        self.Ly = self.param("Y-axis Mesh").names['Length Y']
        self.ny = self.param("Y-axis Mesh").names['Number of y nodes']
        self.delta_y = self.param("Y-axis Mesh").names['Delta y']

        self.xmin.sigValueChanged.connect(self.x_range_Changed)
        self.xmax.sigValueChanged.connect(self.x_range_Changed)
        self.Lx.sigValueChanged.connect(self.Lx_Changed)
        self.nx.sigValueChanged.connect(self.nx_Changed)
        self.delta_x.sigValueChanged.connect(self.delta_x_Changed)

        self.ymin.sigValueChanged.connect(self.y_range_Changed)
        self.ymax.sigValueChanged.connect(self.y_range_Changed)
        self.Ly.sigValueChanged.connect(self.Ly_Changed)
        self.ny.sigValueChanged.connect(self.ny_Changed)
        self.delta_y.sigValueChanged.connect(self.delta_y_Changed)

    def x_range_Changed(self):
        self.Lx.setValue(self.xmax.value() - self.xmin.value(), blockSignal=self.Lx_Changed)
        self.delta_x.setValue(self.Lx.value() / (self.nx.value() - 1),
                              blockSignal=self.delta_x_Changed)
        self.kx_nyquist.setValue(pi / self.delta_x.value())
        self.kx_max.setLimits((0, self.kx_nyquist.value()))

    def Lx_Changed(self):
        self.xmax.setValue(self.xmin.value() + self.Lx.value(), blockSignal=self.x_range_Changed)
        self.delta_x.setValue(self.Lx.value() / (self.nx.value() - 1),
                              blockSignal=self.delta_x_Changed)
        self.kx_nyquist.setValue(pi / self.delta_x.value())
        self.kx_max.setLimits((0, self.kx_nyquist.value()))

    def nx_Changed(self):
        self.delta_x.setValue(self.Lx.value() / (self.nx.value() - 1),
                              blockSignal=self.delta_x_Changed)
        self.kx_nyquist.setValue(pi / self.delta_x.value())
        self.kx_max.setLimits((0, self.kx_nyquist.value()))

    def delta_x_Changed(self):
        self.Lx.setValue(self.delta_x.value() * (self.nx.value() - 1), blockSignal=self.Lx_Changed)
        self.xmax.setValue(self.xmin.value() + self.Lx.value(), blockSignal=self.x_range_Changed)
        self.kx_nyquist.setValue(pi / self.delta_x.value())
        self.kx_max.setLimits((0, self.kx_nyquist.value()))
        self.max_lambda_x.setValue(2 * pi / self.delta_kx.value())

    def kx_range_Changed(self):
        self.delta_kx.setValue(
            (self.kx_max.value() - self.kx_min.value()) / (self.n_kx_nodes.value() - 1),
            blockSignal=self.delta_kx_Changed)
        self.max_lambda_x.setValue(2 * pi / self.delta_kx.value())

    def n_kx_nodes_Changed(self):
        self.delta_kx.setValue(
            (self.kx_max.value() - self.kx_min.value()) / (self.n_kx_nodes.value() - 1),
            blockSignal=self.delta_kx_Changed)
        self.max_lambda_x.setValue(2 * pi / self.delta_kx.value())

    def delta_kx_Changed(self):
        self.kx_max.setValue(
            self.kx_min.value() + (self.n_kx_nodes.value() - 1) * self.delta_kx.value(),
            blockSignal=self.kx_range_Changed)
        self.max_lambda_x.setValue(2 * pi / self.delta_kx.value())

    def y_range_Changed(self):
        self.Ly.setValue(self.ymax.value() - self.ymin.value(), blockSignal=self.Ly_Changed)
        self.delta_y.setValue(self.Ly.value() / (self.ny.value() - 1),
                              blockSignal=self.delta_y_Changed)
        self.ky_nyquist.setValue(pi / self.delta_y.value())
        self.ky_max.setLimits((0, self.ky_nyquist.value()))

    def Ly_Changed(self):
        self.ymax.setValue(self.ymin.value() + self.Ly.value(), blockSignal=self.y_range_Changed)
        self.delta_y.setValue(self.Ly.value() / (self.ny.value() - 1),
                              blockSignal=self.delta_y_Changed)
        self.ky_nyquist.setValue(pi / self.delta_y.value())
        self.ky_max.setLimits((0, self.ky_nyquist.value()))

    def ny_Changed(self):
        self.delta_y.setValue(self.Ly.value() / (self.ny.value() - 1),
                              blockSignal=self.delta_y_Changed)
        self.ky_nyquist.setValue(pi / self.delta_y.value())
        self.ky_max.setLimits((0, self.ky_nyquist.value()))

    def delta_y_Changed(self):
        self.Ly.setValue(self.delta_y.value() * (self.ny.value() - 1), blockSignal=self.Ly_Changed)
        self.ymax.setValue(self.ymin.value() + self.Ly.value(), blockSignal=self.y_range_Changed)
        self.ky_nyquist.setValue(pi / self.delta_y.value())
        self.ky_max.setLimits((0, self.ky_nyquist.value()))
        self.max_lambda_y.setValue(2 * pi / self.delta_ky.value())

    def ky_range_Changed(self):
        self.delta_ky.setValue(
            (self.ky_max.value() - self.ky_min.value()) / (self.n_ky_nodes.value() - 1),
            blockSignal=self.delta_ky_Changed)
        self.max_lambda_y.setValue(2 * pi / self.delta_ky.value())

    def n_ky_nodes_Changed(self):
        self.delta_ky.setValue(
            (self.ky_max.value() - self.ky_min.value()) / (self.n_ky_nodes.value() - 1),
            blockSignal=self.delta_ky_Changed)
        self.max_lambda_y.setValue(2 * pi / self.delta_ky.value())

    def delta_ky_Changed(self):
        self.ky_max.setValue(
            self.ky_min.value() + (self.n_ky_nodes.value() - 1) * self.delta_ky.value(),
            blockSignal=self.ky_range_Changed)
        self.max_lambda_y.setValue(2 * pi / self.delta_ky.value())


class TemporalParameters(pTypes.GroupParameter):
    # create a group of parameters of the time span and resolution which are all dependending on
    # each other so that they should mutually change
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True

        self.logger = logging.getLogger(__name__)

        pTypes.GroupParameter.__init__(self, **opts)

        # the members of the group as they appear in the menu
        self.addChild(
            dict(name='Current Time', type='float', value=0, step=1, siPrefix=True, suffix='s'))
        self.addChild(dict(name='Current Index', type='int', value=0, limits=(0, None)))
        self.addChild(dict(name='Start', type='float', value=0, step=1, siPrefix=True, suffix='s'))
        self.addChild(dict(name='End', type='float', value=60, step=10, siPrefix=True, suffix='s'))
        self.addChild(
            dict(name='Duration', type='float', value=60, step=10, limits=(0.1, None),
                 siPrefix=True, suffix='s'))
        self.addChild(dict(name='Number of samples', type='int', value=600, limit=(1, None)))
        self.addChild(
            dict(name='Delta t', value=0.1, type='float', siPrefix=True, suffix='s', step=1,
                 limits=(0.01, None)))

        self.current_time = self.param('Current Time')
        self.current_index = self.param('Current Index')
        self.t_start = self.param('Start')
        self.t_end = self.param('End')
        self.duration = self.param('Duration')
        self.n_time = self.param('Number of samples')
        self.delta_t = self.param('Delta t')

        self.current_time.sigValueChanged.connect(self.current_time_Changed)
        self.current_index.sigValueChanged.connect(self.current_index_Changed)
        self.t_start.sigValueChanged.connect(self.t_range_Changed)
        self.t_end.sigValueChanged.connect(self.t_range_Changed)
        self.duration.sigValueChanged.connect(self.duration_Changed)
        self.n_time.sigValueChanged.connect(self.n_time_Changed)
        self.delta_t.sigValueChanged.connect(self.delta_t_Changed)

    def current_time_Changed(self):
        self.current_index.setValue(
            int(round((self.current_time.value() - self.t_start.value()) / self.delta_t.value())),
            blockSignal=self.current_index_Changed)

    def current_index_Changed(self):
        self.current_time.setValue(
            self.current_index.value() * self.delta_t.value() + self.t_start.value(),
            blockSignal=self.current_time_Changed)

    def t_range_Changed(self):
        self.duration.setValue(self.t_end.value() - self.t_start.value(),
                               blockSignal=self.duration_Changed)
        self.n_time.setValue(self.duration.value() / self.delta_t.value(),
                             blockSignal=self.n_time_Changed)

    def duration_Changed(self):
        self.t_end.setValue(self.t_start.value() + self.duration.value(),
                            blockSignal=self.t_range_Changed())
        self.n_time.setValue(self.duration.value() / self.delta_t.value(),
                             blockSignal=self.n_time_Changed)

    def n_time_Changed(self):
        self.t_end.setValue(self.t_start.value() + self.delta_t.value() * self.n_time.value(),
                            blockSignal=self.t_range_Changed())
        self.duration.setValue(self.t_end.value() - self.t_start.value(),
                               blockSignal=self.duration_Changed)

    def delta_t_Changed(self):
        self.n_time.setValue(self.duration.value() / self.delta_t.value(),
                             blockSignal=self.n_time_Changed)


class JonSwapParameters(pTypes.GroupParameter):
    # create a group of parameters of the time span and resolution which are all dependending on
    # each other so that they should mutually change
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True

        self.logger = logging.getLogger(__name__)

        pTypes.GroupParameter.__init__(self, **opts)

        # the members of the group as they appear in the menu
        self.addChild(dict(name='Two-D Wave', type='bool', value=False))
        self.addChild(
            dict(name='Significant Wave Height Hs', type='float', value=3.0, step=.1,
                 limits=(0, 100), siPrefix=True,
                 suffix='m'))
        self.addChild(
            dict(name='Peak Period', type='float', value=10, step=1, limits=(0.01, 100),
                 siPrefix=True, suffix='s'))
        self.addChild(
            dict(name='Peak Width', type='float', value=0.0625, step=0.0005, limits=(0, 100),
                 siPrefix=True,
                 suffix='rads/',
                 tip="Width of the Gauss spectrum. Only used for spectrum_type==0."))
        self.addChild(
            dict(name='Peak Enhancement', type='float', value=3.3, step=0.1, limits=(0.1, 10)))
        self.addChild(dict(name='Theta 0', type='float', value=0, step=10, limits=(0, 180),
                           tip="Only waves traveling to the right allowed"))
        self.addChild(dict(name='Theta s spread', type='float', value=5, step=1, limits=(0, None),
                           tip="The s spreading factor should be given. Typical values are 5 wave "
                               "wind wave and 12 for swell. Higher values give less spreading"))

        self.addChild(
            dict(name='Spectrum Type', type='list', values=dict(Jonswap="jonswap", Gauss="gauss"),
                 value="jonswap",
                 tip="Type of spectrum to use: Jonswap (wind waves) or Gauss (swell). Note that for"
                     " the version 0 Gauss spectrum also a width of the spectrum needs to be "
                     "specified."))
        self.addChild(
            dict(name='Spectrum Version', type='list', values=dict(DNV="dnv", HMC="hmc"),
                 value="hmc",
                 tip="The version of implementation to be used for the spectrum type. DNV is the "
                     "standard DNV version, whereas HMC is the matlab implementation"))

        self.twoD = self.param('Two-D Wave')
        self.Hs = self.param('Significant Wave Height Hs')
        self.Tp = self.param('Peak Period')
        self.sigma_gauss = self.param('Peak Width')
        self.gamma = self.param('Peak Enhancement')
        self.theta0 = self.param('Theta 0')
        self.theta_s_spreading_factor = self.param('Theta s spread')
        self.spectrum_type = self.param('Spectrum Type')
        self.spectrum_version = self.param('Spectrum Version')


class WaveParameters(pTypes.GroupParameter):
    # create a group of parameters of the time span and resolution which are all dependending on
    # each other so that they should mutually change
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True

        self.logger = logging.getLogger(__name__)

        pTypes.GroupParameter.__init__(self, **opts)

        # the members of the group as they appear in the menu
        self.addChild(dict(name='Update and Show', type='bool', value=True,
                           tip="Calculate this wave every time  step and show it"))
        self.addChild(dict(name='Save 2D movie frames', type='bool', value=False,
                           tip="Set the filebase name in the movie save dialog"))
        self.addChild(dict(name='Save 2D amplitude data', type='bool', value=False,
                           tip="Set the filebase name in the movie save dialog"))
        self.addChild(dict(name='Scatter Points', type='int', value=5, step=1, limits=(0, 10),
                           tip="Add Scatter Points at the node locations"))
        self.addChild(dict(name='Line Width', type='int', value=1, step=1, limits=(0, 10)))
        self.addChild(
            dict(name='Color', type='color', value="FFFFFF",
                 tip="Choose the color of line plot of this wave"))
        self.scatter_points = self.param('Scatter Points')
        self.update_and_show = self.param('Update and Show')
        self.save_movie_frames = self.param('Save 2D movie frames')
        self.save_amplitude_data = self.param('Save 2D amplitude data')
        self.color = self.param('Color')
        self.width = self.param('Line Width')

        # this are the solver settings

        self.addChild(
            dict(name='Wave Construction', type='list',
                 values=dict(FFT=0, DFTpolar=1, DFTcartesian=2), value=0,
                 tip="Algorithm to reconstruct the wave from the Fourier components. Direct using "
                     "the polar (DFTpolar) or cartesian (DFTcartesian) coordinates or using FFT. "
                     "The latter restricts the number of x and k nodes to a power of 2"))
        self.addChild(dict(name='Wave Selection', type='list',
                           values=dict(All=0, Subrange=1, EqualEnergyBins=2, OneWave=3), value=0,
                           tip="The wave components of the second wave a a selection of the first "
                               "wave. The selection methods are All (all waves are selection), "
                               "Subrange (only the waves between the lower and higher energy limit "
                               "are selected), EqualEnergyBin (the energy per bin is keps equal)"
                               " and OneWave (only one wave is picked)"))
        self.addChild(dict(name='Phase Seed', type='int', value=1, limits=(0, None),
                           tip="The seed of the random generator. For seed=0, a random seed is "
                               "taken"))
        self.addChild(dict(name="Subrange Settings", type="group", children=[
            dict(name='Energy Limit Low', type='float', value=1.0, step=0.1, suffix='%',
                 limits=(0, 99)),
            dict(name='Energy Limit High', type='float', value=90.0, step=1.0, suffix='%',
                 limits=(0, 100)),
            dict(name='Sample Every', type='int', value=1, limits=(1, None),
                 tip="Pick every 'N' nodes"),
            dict(name='Percentage Spreading Area', type='float', value=99.0, limits=(0.00, 100),
                 tip="Percentage to take from area of the spreading function")
        ]))
        self.addChild(dict(name="EqualEnergyBins Settings", type="group", children=[
            dict(name='Number of energy bins', type='int', value=40, limit=(1, None)),
            dict(name='Use Subrange energy limits', type='bool', value=True,
                 tip="Only the k-nodes within the energy range given by the Subrange settings are "
                     "taken"),
            dict(name='Lock nodes to Wave 1', type='bool', value=True,
                 tip="The wave nodes do not coincide with the first wave when initially constructed"
                     ". This means that new" " random seeds are picked. If True, the nearest wave "
                     "node of the first wave is selected such that the waves can be compared")
        ]))
        self.addChild(dict(name="OneWave Settings", type="group", children=[
            dict(name='Wave number index', type='int', value=2, limit=(0, None))
        ]))
        self.wave_construction = self.param('Wave Construction')
        self.wave_selection = self.param('Wave Selection')
        self.E_limit_low = self.param('Subrange Settings').names["Energy Limit Low"]
        self.E_limit_high = self.param('Subrange Settings').names["Energy Limit High"]
        self.theta_area_percentage = self.param('Subrange Settings').names[
            "Percentage Spreading Area"]
        self.sample_every = self.param('Subrange Settings').names["Sample Every"]
        self.seed = self.param("Phase Seed")
        self.picked_wave_index = self.param('OneWave Settings').names['Wave number index']
        self.n_bins_equal_energy = self.param("EqualEnergyBins Settings").names[
            'Number of energy bins']
        self.lock_nodes_to_wave_one = self.param("EqualEnergyBins Settings").names[
            'Lock nodes to Wave 1']
        self.use_subrange_energy_limits = self.param("EqualEnergyBins Settings").names[
            'Use Subrange energy limits']


# create the data containing all data
class JSParameters(object):
    def __init__(self, transfer_parameters=None):

        # initialse the logger
        self.logger = logging.getLogger(__name__)

        self.name = 'Wave Parameters'

        # just the definition of the default parameters of the wave model
        params = [
            JonSwapParameters(name="Jonswap"),
            DomainParameters(name="Domain"),
            TemporalParameters(name="Time"),
            WaveParameters(name="Wave 1"),
            WaveParameters(name="Wave 2")
        ]

        self.Parameters = Parameter.create(name=self.name, type='group', children=params)

    def connect_the_whole_family(self, children, event):
        # recursively connect all the children, childeren of children, etc to the event
        for child in children:
            child.sigValueChanging.connect(event)
            self.connect_the_whole_family(child, event)

    def connect_signals_to_slots(self):

        # first connect all children, grandchilderen etc to the valueChanging event
        self.connect_the_whole_family(self.Parameters.children(), self.valueChanging)

        # conncect the global tree change to the change event
        self.Parameters.sigTreeStateChanged.connect(self.changeTreeState)

    def saveConfiguration(self, fn):
        try:
            state = self.Parameters.saveState()
            pg.configfile.writeConfigFile(state, fn)
        except IOError as e:
            raise Exception(
                "saveConfiguration failed with I/O Error({}): {}".format(e.errno, e.strerror))
        except ValueError as e:
            raise Exception(
                "saveConfiguration failed with Value Error({}): {}".format(e.errno, e.strerror))
        except:
            raise Exception("saveConfiguration failed with Unidentified error")

        return True

    def loadConfiguration(self, fn):
        try:
            state = pg.configfile.readConfigFile(fn)
            self.loadState(state)
        except ParseError as e:
            raise Exception("loadConfiguration failed with ParseError {}".format(e))
        except IOError as e:
            raise Exception(
                "loadConfiguration failed with IOerror ({}) : {}".format(e.errno, e.strerror))
        except Exception as e:
            raise Exception("loadConfiguration failed with Unknown Exception: {}".format(e))

        return True

    def loadState(self, state):
        try:
            # self.Parameters.param(self.name).clearChildren()
            self.Parameters.restoreState(state, removeChildren=False)
            self.connect_signals_to_slots()
        except:
            raise Exception("Failed restoring state")

    def valueChanging(self, param, value):
        self.logger.debug("Value changing (not finalized): {} {}".format(param, value))

    def changeTreeState(self, param, changes):
        self.logger.debug("tree changes:")
        for param, change, data in changes:
            path = self.Parameters.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            self.logger.debug("  parameter: {}".format(childName))
            self.logger.debug("  change:    {}".format(change))
            self.logger.debug("  data:      {}".format(str(data)))
            self.logger.debug("  ----------")
        self.Parameters.emit(QtCore.SIGNAL("tree_parameter_changed"), childName)

    def save(self):
        self.state = self.Parameters.saveState()

    def restoreDefaultState(self):
        self.Parameters.restoreState(self.defaults_state, removeChildren=False)

    def setCurrentState(self):
        self.Parameters.restoreState(self.current_state, removeChildren=False)

    def treeChanged(self, *args):
        self.logger.debug("signal tree changed")
