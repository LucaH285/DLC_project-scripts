# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 19:52:08 2021

@author: Desktop
"""
import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
import seaborn as sns
import time

class InitializeVars:
    def __init__(self):
        self.Source = r''
        self.BodyParts = []
        self.CutOff = 0
        
    def populate(self, Path, BodyPartList, PValCutOff):
        self.Source = Path
        self.BodyParts = BodyPartList
        self.CutOff = PValCutOff

class FileImport:
    def ImportFunction_IfPath(self, Source):
        CSVFileList = []
        for files in os.listdir(Source):
            if files.endswith(".csv"):
                CSVFileList.append(pd.read_csv(files))
        return(CSVFileList)
                    
    def ImportFunction_IfFile(self, Source):
        CSVFileList = [pd.read_csv(Source)]
        return(CSVFileList)
    
class FileExport:
    """
    Export to same location as import
    """
    def ExportFunction(self, Frame, Source, Name):
        ExportSource = "{0}/{1}".format(Source, Name)
        Frame.to_csv(ExportSource)

class initVars(ABC):
    @abstractmethod
    def VarLoads(self, Init):
        pass
    
class Imports(ABC):
    @abstractmethod
    def InheritImport(self, importFxn):
        pass

class Export(ABC):
    def InheritExport(self):
        pass
    
class ComputeEuclideanDistance(initVars, Imports):
    def InheritImport(self, Import, ImportSource):
        if os.path.isdir(ImportSource) is True:
            return(importFxn.ImportFunction_IfPath(ImportSource))
        elif os.path.isdir(ImportSource) is False:
            return(importFxn.ImportFunction_IfFile(ImportSource))
    
    def preprocessFrame(self, InputDataFrame):
        ResetColNames = {
            InputDataFrame.columns.values[Int]:Int for Int in range(len(InputDataFrame.columns.values))
            }
        InputDataFrame = InputDataFrame.rename(columns=ResetColNames).drop([0], axis = 1)
        """
        Retrieve body parts .iloc[row, column]
        """
        BodyParts = list(set(InputDataFrame.iloc[0,]))
        BodyParts = [Names for Names in BodyParts if Names != "bodyparts"]
        TrimmedFrame = InputDataFrame.iloc[2:,]
        return(TrimmedFrame, BodyParts)
    
    def checkPvals(self, InputDataFrame, CutOff):
        for Cols in InputDataFrame.columns.values:
            if Cols % 3 == 0:
                for Vals in range(len(InputDataFrame[Cols])):
                    if Vals < CutOff:
                        
                        
                    
    
    def VarLoads(self, Init):
        DLCFrame = self.InheritImport(importFxn, ImportSource=Init.Source)
        DLCFrames = [self.preprocessFrame(InputDataFrame=Frames) for Frames in DLCFrame]
        for Frames in DLCFrames:
            self.checkPvals(Frames[0])
        return(DLCFrames)
        
        
        
        



if __name__ == '__main__':
    Init = InitializeVars()
    Init.populate(Path=r"F:\work\20191205-20200507T192029Z-001\20191205\162658_480x360DeepCut_resnet50_RatNov29shuffle1_1030000.csv",
                  BodyPartList=["nose", "head", "body", "tail"], PValCutOff=0.95)
    
    importFxn = FileImport()
    # importFxn.ImportFunction_IfFile(Source=r"F:\work\20191205-20200507T192029Z-001\20191205\162658_480x360DeepCut_resnet50_RatNov29shuffle1_1030000.csv")
    ComputeEuclideanDistance().VarLoads(Init)

        
        
    
            