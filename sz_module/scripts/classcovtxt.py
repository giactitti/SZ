# -*- coding: utf-8 -*-

"""
/***************************************************************************
    02 FR Fitting/CrossValid
        begin                : 2021-11
        copyright            : (C) 2021 by Giacomo Titti,
                               Padova, November 2021
        email                : giacomotitti@gmail.com
 ***************************************************************************/

/***************************************************************************
    02 FR Fitting/CrossValid
    Copyright (C) 2021 by Giacomo Titti, Padova, November 2021

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 ***************************************************************************/
"""

__author__ = 'Giacomo Titti'
__date__ = '2021-07-01'
__copyright__ = '(C) 2021 by Giacomo Titti'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from qgis.PyQt.QtCore import QCoreApplication,QVariant
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterRasterLayer,
                       QgsMessageLog,
                       Qgis,
                       QgsProcessingMultiStepFeedback,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsVectorLayer,
                       QgsRasterLayer,
                       QgsProject,
                       QgsField,
                       QgsFields,
                       QgsVectorFileWriter,
                       QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsProcessingParameterField,
                       QgsProcessingParameterString,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterField,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterFile)
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import math
import operator
import matplotlib.pyplot as plt

from qgis import processing
from osgeo import gdal,ogr,osr
import numpy as np
import math
import operator
import random
from qgis import *
# ##############################
import matplotlib.pyplot as plt
import csv
from processing.algs.gdal.GdalUtils import GdalUtils
#import plotly.express as px
#import chart_studio
import plotly.offline
import plotly.graph_objs as go
#import geopandas as gd
import pandas as pd

class classcovtxtAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    STRING = 'STRING'
    FILE = 'FILE'
    STRING3 = 'STRING3'
    OUTPUT = 'OUTPUT'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return classcovtxtAlgorithm()

    def name(self):
        return 'classy filed by file.txt'

    def displayName(self):
        return self.tr('06 Classify field by file.txt')

    def group(self):
        return self.tr('Data preparation')

    def groupId(self):
        return 'Data preparation'

    def shortHelpString(self):
        return self.tr("Apply classification to field from file.txt")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT, self.tr('covariates'), types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterFile(self.FILE, 'Txt classes', QgsProcessingParameterFile.File, '', defaultValue=None))
        self.addParameter(QgsProcessingParameterField(self.STRING, 'field', parentLayerParameterName=self.INPUT, defaultValue=None))
        self.addParameter(QgsProcessingParameterString(self.STRING3, 'new field name', defaultValue=None))

        #self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, 'Output test',fileFilter='GeoPackage (*.gpkg *.GPKG)', defaultValue=None))

        #self.addParameter(QgsProcessingParameterField(self.STRING3, 'weight', parentLayerParameterName=self.INPUT, defaultValue=None))

        #self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT1, self.tr('LSI'), defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT1, 'edgesJenks', '*.txt', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT2, 'edgesEqual', '*.txt', defaultValue=None))
        #self.addParameter(QgsProcessingParameterFile(self.OUTPUT3, 'edges', '*.txt', defaultValue=None))
        #self.addParameter(QgsProcessingParameterNumber(self.NUMBER, self.tr('number of classes'), type=QgsProcessingParameterNumber.Integer, defaultValue = None,  minValue=0))
        #self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT2, self.tr('Landslides'), types=[QgsProcessing.TypeVectorPoint], defaultValue=None))

    def processAlgorithm(self, parameters, context, model_feedback):
        #parameters['classes']=5
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(1, model_feedback)
        results = {}
        outputs = {}


        source = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        parameters['covariates']=source.source()
        if parameters['covariates'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        parameters['field'] = self.parameterAsString(parameters, self.STRING, context)
        if parameters['field'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING))

        parameters['txt'] = self.parameterAsFile(parameters, self.FILE, context)#.source()
        if parameters['txt'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.FILE))
        print(parameters['txt'])

        parameters['nome'] = self.parameterAsString(parameters, self.STRING3, context)
        if parameters['nome'] is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.STRING3))

        # parameters['out'] = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        # if parameters['out'] is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.OUTPUT))

        alg_params = {
        'INPUT_VECTOR_LAYER': parameters['covariates'],
        'field': parameters['field'],
        'txt' : parameters['txt'],
        'nome' : parameters['nome']
            }

        outputs['crs']=self.classify(alg_params)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        return results

    def classify(self,parameters):###############classify causes according to txt classes
        Min={}
        Max={}
        clas={}
        with open(parameters['txt'], 'r') as f:
            c = csv.reader(f,delimiter=' ')
            countr=1
            for cond in c:
                b=np.array([])
                b=np.asarray(cond)
                Min[countr]=b[0].astype(np.float32)
                Max[countr]=b[1].astype(np.float32)
                clas[countr]=b[2]#.astype(int)
                countr+=1
        key_max=None
        key_min=None
        key_max = max(Max.keys(), key=(lambda k: Max[k]))
        key_min = min(Min.keys(), key=(lambda k: Min[k]))




        #layer = QgsVectorLayer('Polygon?crs=EPSG:4326', 'poly', 'memory')


        #poly = QgsFeature()
        #fields = layer.pendingFields()
        layer = QgsVectorLayer(parameters['INPUT_VECTOR_LAYER'], '', 'ogr')
        crs=layer.crs()

        pr = layer.dataProvider()
        attr = pr.addAttributes([QgsField(parameters['nome'], QVariant.Int)])
        layer.updateFields()
        layer.startEditing()
        features = layer.getFeatures()
        count=0
        feat=[]
        attr=np.array([])
        for feature in features:
            #field=np.vstack((attr,feature.attribute(parameters['field'])))
            ff=feature.attribute(parameters['field'])
            for i in range(1,countr):
                if ff>=Min[i] and ff<Max[i]:
                    #print(type(int(clas[i])))
                    #ff(ff>=Min[i])&(ff<Max[i])]=clas[i]
                    #feature.setAttribute(parameters['nome'], clas[i])
                    feature[parameters['nome']]=int(clas[i])
                    layer.updateFeature(feature)
        #print(ff,'ff')
        layer.commitChanges()
        QgsProject.instance().reloadAllLayers()

        #idx=np.where(np.isnan(self.matrix))
        #self.matrix[idx]=-9999
        #self.RasterInt[idx]=-9999
        #self.matrix[(self.matrix<Min[key_min])]=-9999
        #self.RasterInt[(self.RasterInt<Min[key_min])]=-9999
        #self.matrix[(self.matrix>Max[key_max])]=-9999
        #self.RasterInt[(self.RasterInt>Max[key_max])]=-9999
        #del self.RasterInt

        #self.matrix2=np.zeros(np.shape(self.matrix),dtype='float32')
        #self.matrix2[:]=self.matrix[:]
        #self.matrix2[self.matrix2==0]=-9999

        return(crs)

    # def save(self,parameters):
    #     # define fields for feature attributes. A QgsFields object is needed
    #     fields = QgsFields()
    #
    #     fields.append(QgsField('ID', QVariant.Int))
    #
    #     for field in nomi:
    #         if field=='ID':
    #             fields.append(QgsField(field, QVariant.Int))
    #         if field=='geom':
    #             continue
    #         if field=='y':
    #             fields.append(QgsField(field, QVariant.Int))
    #         else:
    #             fields.append(QgsField(field, QVariant.Double))
    #
    #     #crs = QgsProject.instance().crs()
    #     transform_context = QgsProject.instance().transformContext()
    #     save_options = QgsVectorFileWriter.SaveVectorOptions()
    #     save_options.driverName = 'GPKG'
    #     save_options.fileEncoding = 'UTF-8'
    #
    #     writer = QgsVectorFileWriter.create(
    #       parameters['OUT'],
    #       fields,
    #       QgsWkbTypes.Polygon,
    #       parameters['crs'],
    #       transform_context,
    #       save_options
    #     )
    #
    #     if writer.hasError() != QgsVectorFileWriter.NoError:
    #         print("Error when creating shapefile: ",  writer.errorMessage())
    #     for i, row in df.iterrows():
    #         fet = QgsFeature()
    #         fet.setGeometry(QgsGeometry.fromWkt(row['geom']))
    #         fet.setAttributes(list(map(float,list(df.loc[ i, df.columns != 'geom']))))
    #         writer.addFeature(fet)
    #
    #     # delete the writer to flush features to disk
    #     del writer
