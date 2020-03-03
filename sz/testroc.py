import numpy as np
import matplotlib.pyplot as plt
#from itertools import cycle
# from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# #from scipy import interp
from sklearn.metrics import roc_auc_score
#import numpy as np
from osgeo import gdal,osr,ogr
import sys
import math
import csv
from qgis.core import QgsMessageLog
import os
import operator
#import operator

class classifier:
    def input(self):
        self.ds2 = gdal.Open(self.lsi)#A
        a=self.ds2.GetRasterBand(1)
        NoData=a.GetNoDataValue()
        self.matrix = np.array(a.ReadAsArray()).astype(float)
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
        #print(w,h,xmin,xmax,ymin,ymax,self.xsize,self.ysize)

        # self.ds = gdal.Open(self.inv)#A
        # b=self.ds.GetRasterBand(1)
        # NoData=b.GetNoDataValue()
        b=self.vector2array(self.inv,w,h,xmin,ymin,xmax,ymax,self.xsize,self.ysize)
        self.inventory = b.astype(int)

        # self.ds = gdal.Open(self.val)#A
        # c=self.ds.GetRasterBand(1)
        # NoData=c.GetNoDataValue()
        c=self.vector2array(self.val,w,h,xmin,ymin,xmax,ymax,self.xsize,self.ysize)
        self.validation = c.astype(int)

        # self.ds = gdal.Open(self.train)#A
        # d=self.ds.GetRasterBand(1)
        # NoData=d.GetNoDataValue()
        d=self.vector2array(self.train,w,h,xmin,ymin,xmax,ymax,self.xsize,self.ysize)
        self.training = d.astype(int)

        self.numOff=100#divisibile per 5
        self.Off=20

    def classy(self):
        l=self.xsize*self.ysize
        self.matrix=np.reshape(self.matrix, l)
        self.inventory=np.reshape(self.inventory, l)
        idx=np.where(self.matrix==-9999.)
        self.scores = np.delete(self.matrix,idx)
        self.y_scores=np.delete(self.matrix,idx)
        self.y_true = np.delete(self.inventory,idx)
        self.y_v = np.delete(self.validation,idx)
        self.y_t = np.delete(self.training,idx)
        nclasses=5
        M=np.max(self.scores)
        m=np.min(self.scores)
        count=0
        ran=np.array([])
        fitness=0
        values=np.array([])
        classes=([])
        c={}
        ran=np.array([])
        summ=0
        while count<self.Off:
            weight={}
            fpr={}
            tpr={}
            tresh={}
            roc_auc={}
            ran=np.array([])
            FPR={}
            TPR={}
            mm=None
            if count==0:
                c={}
                for pop in range(self.numOff):
                    ran=np.sort(np.random.random_sample(nclasses-1)*(M-m))
                    c[pop]=np.hstack((m,m+ran,M+1))
                    #print c
                    #print ciao
            else:
                c=file
            for k in range(self.numOff):
                #print weight,'weight'
                weight[k]=self.y_scores
                for i in range(nclasses):
                    index=np.array([])
                    index=np.where((self.scores>=c[k][i]) & (self.scores<c[k][i+1]))
                    weight[k][index]=float(i+1)
                #roc_auc[k]=roc_auc_score(self.y_true, weight[k], None)
                ####################################
                fpr = {}
                tpr = {}
                P=float(len(np.argwhere(self.y_true==1)))#tp+fn
                N=float(len(np.argwhere(self.y_true==0)))#tn+fp
                for i in range(nclasses):
                    index=np.array([])
                    fptp=np.array([])
                    index=np.where(weight[k]==i+1)
                    fptp=self.y_true[index]
                    tp=float(len(np.argwhere(fptp==1)))
                    fp=float(len(np.argwhere(fptp==0)))
                    #print tp,':tp',fp,':fp'
                    #print P,':P',N,':N'
                    fpr[i]=float(fp/N)
                    tpr[i]=float(tp/P)
                    #FPR[i]=fpr[i]
                    #TPR[i]=tpr[i]
                FPR[k]=np.array([0,fpr[4],fpr[4]+fpr[3],fpr[4]+fpr[3]+fpr[2],fpr[4]+fpr[3]+fpr[2]+fpr[1],fpr[4]+fpr[3]+fpr[2]+fpr[1]+fpr[0]])
                #TPR=np.array([tpr[0],tpr[0]+tpr[1],tpr[0]+tpr[1]+tpr[2],tpr[0]+tpr[1]+tpr[2]+tpr[3],tpr[0]+tpr[1]+tpr[2]+tpr[3]+tpr[4]])
                TPR[k]=np.array([0,tpr[4],tpr[4]+tpr[3],tpr[4]+tpr[3]+tpr[2],tpr[4]+tpr[3]+tpr[2]+tpr[1],tpr[4]+tpr[3]+tpr[2]+tpr[1]+tpr[0]])
                roc_auc[k]=np.trapz(TPR[k],FPR[k])
            ###############################################
            mm=None
            mm=max(roc_auc, key=roc_auc.get)
            #print fitness
            if roc_auc[mm]>fitness:#############################fitness
                fitness=None
                classes=np.array([])
                values=np.array([])
                ttpr=np.array([])
                ffpr=np.array([])
                fitness=roc_auc[mm]
                print(fitness,'AUC')
                classes=c[mm]
                values=weight[mm]
                print(classes,'Classes')
                ttpr=TPR[mm]
                ffpr=FPR[mm]
                summ=1
            else:
                summ+=1
            ##########################PASS
            #print(count)
            count+=1

            #if summ==4:
                #count=self.Off
                #print 'stop'
            #########################GA
            file={}
            qq=0
            for q in range(0,self.numOff,5):
                #print q,'qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq'
                a=np.array([])
                bb=[]
                cc=[]
                cc=list(roc_auc.items())
                #print(cc[q:q+5],'qui')
                bb=dict(cc[q:q+5])
                a=sorted(bb.items(), key=operator.itemgetter(1),reverse=True)
                file[q]=c[a[0][0]]
                #file[qq+1]=np.hstack((file[qq][0],np.random.uniform(file[qq][0],file[qq][1],1),file[qq][2:]))
                #file[qq+2]=np.hstack((file[qq][:2],np.random.uniform(file[qq][1],file[qq][2],1),file[qq][3:]))
                #file[qq+3]=np.hstack((file[qq][:3],np.random.uniform(file[qq][2],file[qq][3],1),file[qq][4:]))
                #file[qq+4]=np.hstack((file[qq][:4],np.random.uniform(file[qq][3],file[qq][4],1),file[qq][5:]))
                file[q+1]=np.hstack((file[q][0],file[q][0]+(np.sort(np.random.random_sample(1)*(file[q][2]-file[q][0]))),file[q][2:]))
                file[q+2]=np.hstack((file[q][:2],file[q][1]+(np.sort(np.random.random_sample(1)*(file[q][3]-file[q][1]))),file[q][3:]))
                file[q+3]=np.hstack((file[q][:3],file[q][2]+(np.sort(np.random.random_sample(1)*(file[q][4]-file[q][2]))),file[q][4:]))
                file[q+4]=np.hstack((file[q][:4],file[q][3]+(np.sort(np.random.random_sample(1)*(file[q][5]-file[q][3]))),file[q][5:]))
                #file[qq+1]=np.hstack((file[qq][:2],file[qq][2]+(np.sort(np.random.random_sample(3)*(file[qq][5]-file[qq][1]))),file[qq][5]))
                #file[qq+2]=np.hstack((file[qq][0],file[qq][0]+(np.sort(np.random.random_sample(1)*(file[qq][2]-file[qq][0]))),file[qq][2],file[qq][2]+(np.sort(np.random.random_sample(2)*(file[qq][5]-file[qq][2]))),file[qq][5]))
                #file[qq+3]=np.hstack((file[qq][0],file[qq][0]+(np.sort(np.random.random_sample(2)*(file[qq][3]-file[qq][0]))),file[qq][3],file[qq][3]+(np.sort(np.random.random_sample(1)*(file[qq][5]-file[qq][3]))),file[qq][5]))
                #file[qq+4]=np.hstack((file[qq][0],file[qq][0]+(np.sort(np.random.random_sample(3)*(file[qq][4]-file[qq][0]))),file[qq][4:]))

                #file[qq+1]=np.hstack((file[qq][:2],np.random.uniform(file[qq][1],file[qq][5],3),file[qq][5]))
                #file[qq+2]=np.hstack((file[qq][0],np.random.uniform(file[qq][0],file[qq][2],1),file[qq][2],np.random.uniform(file[qq][2],file[qq][5],2),file[qq][5]))
                #file[qq+3]=np.hstack((file[qq][0],np.random.uniform(file[qq][0],file[qq][3],2),file[qq][3],np.random.uniform(file[qq][3],file[qq][5],1),file[qq][5]))
                #file[qq+4]=np.hstack((file[qq][0],np.random.uniform(file[qq][0],file[qq][4],3),file[qq][4:]))
                qq+=5
        self.fitness=None
        self.tpr=np.array([])
        self.fpr=np.array([])
        self.values=np.array([])
        self.classes=np.array([])
        self.fitness=fitness
        self.values=values
        self.classes=classes
        self.tpr=ttpr
        self.fpr=ffpr
        file = open(self.fold+'/plotROC.txt','w')#################save txt
        var=[self.fpr,self.tpr]
        file.write('false positive, true positive: %s\n' %var)#################save fp,tp
        np.savetxt(self.out, self.classes, delimiter=',')

    def stamp(self):
        ################################figure
        fpr1, tpr1, tresh1 = roc_curve(self.y_true,self.scores)

        fprv, tprv, treshv = roc_curve(self.y_v,self.scores)
        fprt, tprt, tresht = roc_curve(self.y_t,self.scores)

        #print self.fpr
        #print self.tpr
        #print self.classes
        r=roc_auc_score(self.y_true, self.scores, None)
        plt.figure()
        lw = 4
        plt.plot(fpr1, tpr1, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f)' %r)

        plt.plot(fprv, tprv, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f)' %r)
        plt.plot(fprt, tprt, color='green',lw=lw, label= 'Complete dataset (AUC = %0.2f)' %r)

        plt.plot(self.fpr, self.tpr, 'ro')
        plt.plot(self.fpr, self.tpr, color='darkorange',lw=lw, label='Classified dataset (AUC = %0.2f)' % self.fitness)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()

    # def array2raster(self,newRasterfn,pixelWidth,pixelHeight,array,oo):
    #     cr=np.shape(array)
    #     cols=cr[1]
    #     rows=cr[0]
    #     originX = oo[0]
    #     originY = oo[1]
    #     driver = gdal.GetDriverByName('GTiff')
    #     outRaster = driver.Create(newRasterfn, int(cols), int(rows), 1, gdal.GDT_Float32)
    #     outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    #     outband = outRaster.GetRasterBand(1)
    #     outband.SetNoDataValue(-9999)
    #     outband.WriteArray(array)
    #     outRasterSRS = osr.SpatialReference()
    #     outRasterSRS.ImportFromEPSG(int(self.epsg[self.epsg.rfind(':')+1:]))
    #     outRaster.SetProjection(outRasterSRS.ExportToWkt())
    #     outband.FlushCache()

    def vector2array(self,inn,pxlw,pxlh,xm,ym,xM,yM,sizex,sizey):
        driverd = ogr.GetDriverByName('ESRI Shapefile')
        ds9 = driverd.Open(inn)
        layer = ds9.GetLayer()
        count=0
        for feature in layer:
            count+=1
            geom = feature.GetGeometryRef()
            xy=np.array([geom.GetX(),geom.GetY()])
            try:
                XY=np.vstack((XY,xy))
            except:
                XY=xy
        size=np.array([pxlw,pxlh])
        OS=np.array([xm,yM])
        NumPxl=(np.ceil(abs((XY-OS)/size)-1))#from 0 first cell
        valuess=np.zeros((sizey,sizex),dtype='int16')
        #print(XY)
        #print(NumPxl)
        #print(len(NumPxl))
        #print(count)
        try:
            for i in range(count):
                #print(i,'i')
                if XY[i,1]<yM and XY[i,1]>ym and XY[i,0]<xM and XY[i,0]>xm:
                    valuess[NumPxl[i,1].astype(int),NumPxl[i,0].astype(int)]=1
        except:#only 1 feature
            if XY[1]<yM and XY[1]>ym and XY[0]<xM and XY[0]>xm:
                valuess[NumPxl[1].astype(int),NumPxl[0].astype(int)]=1
        fuori = valuess.astype('float32')
        return fuori
