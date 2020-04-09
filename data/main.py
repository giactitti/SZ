from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterRasterLayer
from qgis.core import QgsProcessingParameterField
from qgis.core import QgsProcessingParameterVectorDestination
from qgis.core import QgsProcessingParameterFileDestination
from qgis.core import QgsRasterLayer
from qgis.core import QgsProcessingParameterExpression
import processing
#import plotly.express as px
import numpy as np
#import chart_studio.plotly as py
#import plotly.plotly as py
#import chart_studio.plotly as py
from PyQt5.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import os.path
##############################
import numpy as np
from osgeo import gdal,osr
import sys
import math
import csv
from qgis.core import QgsMessageLog, QgsProject, QgsRasterLayer
import sys
##############################
import matplotlib.pyplot as plt
from test import *
#############################
import sys,os
from qgis.core import *
from qgis.gui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtGui import *
from qgis.utils import iface

ran=statistic(iface)

parameters={}
parameters['lsi']='/home/irpi/repos/SZ/data/lsiclassified.tif'
parameters['r1']='/home/irpi/repos/SZ/data/reclassedcause0.tif'

ran.processAlgorithm(parameters)