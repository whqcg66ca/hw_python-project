# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:59:07 2022

@author: Wanghongq
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:20:06 2022
@author: Hongquan
https://gdal.org/tutorials/raster_api_tut.html#using-create
https://www.youtube.com/watch?v=p_BsFdV_LUk
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
# os.environ['PROJ_LIB'] = r'C:\Users\Hongquan\anaconda3\pkgs\proj-6.2.1-h3758d61_0\Library\share\proj'
os.environ['PROJ_LIB'] = r'C:\Users\WangHongq\Anaconda3\pkgs\proj-6.2.1-h3758d61_0\Library\share\proj'

# os.environ['GDAL_DATA'] = r'C:\Users\Hongquan\anaconda3\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\WangHongq\Anaconda3\Library\share\proj'

from osgeo import gdal
# from PIL import Image
# import rasterio
# import geopandas as gpd
#import sys

targ='NUECanola'  
 # 'leth_mustard', 'leth_oat'

#Teget1=glob.glob(r'N:\UAV Data_Lethbridge Projects 2024\UAV RedEdge P (MSI)_Processed\Pix4D\SoiResidue_20240510\SoilResidueCoverCrop20240510\4_index\reflectance', recursive = True)
Teget1=glob.glob(r'N:\UAV Data_RSPLab Projects 2025\UAV RedEdge P (MSI)_Processed/' + r'*'+targ+ r'*'+ r'/*/4_index/reflectance', recursive = True)
#%% Read the bands in a folder
# file_list= glob.glob(r'C:\Users\Hongquan\Python_test\Test50_June 30\Reflectance30062022\4_index\reflectance\*.tif') 

# Teget=[Teget1[i] for i in [0,1,4,5]]
Teget=Teget1

for datapath in Teget:
    print(datapath)
    # datapath=r'G:\Lethbridge UAV Data_WGRF 2022 (Keshav)\UAV RedEdge P (MSI)_Processed\Mustard*\Reflectance*'
    add1=r'\*.tif'
    searchpath=datapath+add1
    
    # add2=r'\Mustard-1DAT-06082022.tif'
    # add2= searchpath[71:94]
    
   
    
    ind=searchpath.find(targ)
  
    add2= searchpath[ind-18:ind+17] #23JD, 15LT 19OML 20sw
    datapath2=r'N:\UAV Data_RSPLab Projects 2025\UAV RedEdge P (MSI)_Processed\Mosaic\tmp'
    # datapath2=r'G:\Lacombe UAV Data_WGRF 2022 (Kelly)_processed'
    savepath=datapath2 +  r'\\' +  add2 + r'.tif'
    
    file_list1= sorted(glob.glob(searchpath))
    # order=[0,1,4,3,2] # for lacombe, and saskatoon
    order=[3,0,1,5,4,2] # for lethbridge
    file_list=[file_list1[i] for i in order]
    
    toto=gdal.Open(file_list[0])
    img_width,img_height=toto.RasterXSize,toto.RasterYSize
    
    # binmask2=np.zeros((5781,3560,5))
    binmask2=np.zeros((img_height,img_width,len(file_list)))
    
    i=0
    for fname in file_list:
    
        ds=gdal.Open(fname)
        gt=ds.GetGeoTransform()
        proj=ds.GetProjection()
        
        
        band=ds.GetRasterBand(1)
        Type = ds.GetRasterBand(1).DataType
        myType = gdal.GetDataTypeName(Type)   
        array=band.ReadAsArray()
        
        plt.figure()
        plt.imshow(array,vmin=0, vmax=0.1)
        
        binmask2[:,:,i]=array
        i=i+1
        
        # binmask=np.where((array>=0),0,10)
        # plt.figure()
        # plt.imshow(binmask,vmin=0, vmax=0.5)
        
    #%% Write the geotiff band stack
    
    mydriver=gdal.GetDriverByName("GTiff")
    mydriver.Register()
    # outds=mydriver.Create(r"C:\Users\Hongquan\Python_test\Gdal_test\gdaltest2.tif",\
    #                       xsize=binmask.shape[1],ysize=binmask.shape[0],bands=1,eType=gdal.GDT_CFloat32)   
    # outds=mydriver.Create(r"C:\Users\Hongquan\Python_test\Gdal_test\gdaltest22.tif",\
    #                           xsize=binmask2.shape[1],ysize=binmask2.shape[0],bands=5,eType=gdal.GDT_Float32)
    
    outds=mydriver.Create(savepath,xsize=binmask2.shape[1],ysize=binmask2.shape[0],bands=len(file_list),eType=gdal.GDT_Float32)
        
    for m in range(1,len(file_list)+1,1):   
        outds.SetGeoTransform(gt)
        outds.SetProjection(proj)
        outband=outds.GetRasterBand(m)
        outband.WriteArray(binmask2[:,:,m-1])
        outband.FlushCache()    
         
    outband=None
    outds=None
      
    # outband.SetNoDataValue(np.nan)
     
#%%

# def GetGeoInfo(FileName):
#     # if exists(FileName) is False:
#     #         raise Exception('[Errno 2] No such file or directory: \'' + FileName + '\'')         
#     SourceDS = gdal.Open(FileName, gdal.GA_ReadOnly)
#     # if SourceDS == None:
#     #     raise Exception("Unable to read the data file")    
#     NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
#     xsize = SourceDS.RasterXSize
#     ysize = SourceDS.RasterYSize
#     GeoT = SourceDS.GetGeoTransform()
#     Projection = SourceDS.SpatialReference()
#     Projection.ImportFromWkt(SourceDS.GetProjectionRef())
#     DataType = SourceDS.GetRasterBand(1).DataType
#     DataType = gdal.GetDataTypeName(DataType)   
#     return NDV, xsize, ysize, GeoT, Projection, DataType
