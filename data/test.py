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
#from .test import *
#############################
import sys,os
from qgis.core import *
from qgis.gui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtGui import *
from qgis.utils import iface

class statistic:
    def __init__(self,iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        #self.plugin_dir = os.path.dirname(__file__)
        
    def processAlgorithm(self, parameters):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        outputs={}
        alg_params = {
            'INPUT': parameters['lsi']
        }
        outputs['Mlsi']=self.input(alg_params)

        ##############################################

        # Input
        alg_params = {
            'INPUT': parameters['r1']
        }
        outputs['Mr1']=self.input(alg_params)

        # Stat
        alg_params = {
            'INPUT1': outputs['Mlsi'],
            'INPUT2': outputs['Mr1']
        }
        outputs['indlsi1']=self.stat(alg_params)

        # Frame
        alg_params = {
            'INPUT': outputs['indlsi1'],
            'OUTPUT': '/tmp/fuori.csv'
        }
        self.frame(alg_params)

    def input(self,parameters):
            #QgsMessageLog.logMessage(parameters['lsi'], 'MyPlugin', level=Qgis.Info)
            #raster = QgsRasterLayer(parameters['INPUT'])
            #if not raster.dataProvider().name() == 'gdal':
            #    raise QgsProcessingException(self.tr('This algorithm can only be used with GDAL raster layers'))
            #rasterPath = raster.source()
            self.ds2 = gdal.Open(parameters['INPUT'])
            #QgsMessageLog.logMessage(rasterPath, 'MyPlugin', level=Qgis.Info)
            #layer = QgsRasterLayer(parameters['INPUT'])
            #fileInfo = QFileInfo(parameters['INPUT'])
            #path = fileInfo.filePath()
            #baseName = fileInfo.baseName()
            #layer = QgsRasterLayer(path, baseName)
            #layer.dataProvider().source()
            #QgsMessageLog.logMessage(path, 'MyPlugin', level=Qgis.Info)
            #QgsMessageLog.logMessage(baseName, 'MyPlugin', level=Qgis.Info)
            #QgsMessageLog.logMessage(layer.source(), 'MyPlugin', level=Qgis.Info)
            #self.ds2 = gdal.Open(layer.source())#A
            a=self.ds2.GetRasterBand(1)
            NoData=a.GetNoDataValue()
            matrix = np.array(a.ReadAsArray()).astype(float)
            self.xsize = self.ds2.RasterXSize
            self.ysize = self.ds2.RasterYSize
            #upx, xres, xskew, upy, yskew, yres = self.ds2.GetGeotransform()
            gt=self.ds2.GetGeoTransform()
            w=gt[1]
            h=gt[5]
            xmin=gt[0]
            ymax=gt[3]
            xmax=xmin+(self.xsize*w)
            ymin=ymax+(self.ysize*h)
            return matrix

    def stat(self,parameters):
        lsim=parameters['INPUT1'].astype(int)
        m2=parameters['INPUT2'].astype(int)
        indm2={}
        indlsi={}
        for i in range(1,lsim.max()+1,1):
            for ii in range(1,m2.max()+1,1):
                ind=None
                ind=np.where((lsim==i)&(m2==ii))
                indm2[ii]=len(ind)
            indlsi[i]=indm2
        return indlsi

    def frame(self,parameters):
        d=parameters['INPUT']
        df = pd.DataFrame([(k,k1,v1) for k,v in d.items() for k1,v1 in v.items()], columns = ['lsi','cause','length'])
        df.to_csv(parameters['OUTPUT'], encoding='utf-8', index=False)
