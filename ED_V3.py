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
from collections import OrderedDict
import matplotlib.pyplot as mp

class InitializeVars:
    def __init__(self, Path, BodyPartList, PValCutOff, ExportSource):
        self.Source = Path
        self.BodyParts = BodyPartList
        self.CutOff = PValCutOff
        self.ExportSource = ExportSource

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
    @abstractmethod
    def InheritExport(self):
        pass

class ComputeEuclideanDistance(initVars, Imports, Export):
    def InheritImport(self, Import, ImportSource):
        if os.path.isdir(ImportSource) is True:
            return(importFxn.ImportFunction_IfPath(ImportSource))
        elif os.path.isdir(ImportSource) is False:
            return(importFxn.ImportFunction_IfFile(ImportSource))
        
    def InheritExport(self, ExportFrame, ExportSource, FileName):
        return(FileExport().ExportFunction(Frame = ExportFrame, Source = ExportSource, Name = FileName))
    
    def preprocessFrame(self, InputDataFrame):
        ResetColNames = {
            InputDataFrame.columns.values[Int]:Int for Int in range(len(InputDataFrame.columns.values))
            }
        InputDataFrame = InputDataFrame.rename(columns=ResetColNames).drop([0], axis = 1)
        """
        Retrieve body parts .iloc[row, column]
        """
        BodyParts = list(OrderedDict.fromkeys(list(InputDataFrame.iloc[0,])))
        BodyParts = [Names for Names in BodyParts if Names != "bodyparts"]
        TrimmedFrame = InputDataFrame.iloc[2:,]
        return(TrimmedFrame, BodyParts)

    def checkPvals(self, InputDataFrame, CutOff):
        for Cols in InputDataFrame.columns.values:
            if Cols % 3 == 0:
                for Vals in InputDataFrame.index.values:
                    if float(InputDataFrame[Cols][Vals]) < CutOff:
                        #Using Iloc/loc here like that will take the whole row, all body parts.
                        #You can use the loc[row, column] to specify the columns, this should work
                        XCoordinates = Cols - 3
                        PValScore = Cols
                        PreviousRow = InputDataFrame.loc[Vals - 1, XCoordinates:PValScore]
                        InputDataFrame.loc[Vals, XCoordinates:PValScore] = PreviousRow
        return(InputDataFrame)

    def computeEuclidean(self, InputDataFrame, BodyParts):
        DistanceVectors = [[] for _ in range(len(BodyParts))]
        ColsToDrop = [Cols for Cols in InputDataFrame.columns.values if Cols % 3 == 0]
        InputDataFrame = InputDataFrame.drop(ColsToDrop, axis = 1)
        Counter = 0
        for Cols1, Cols2 in zip(InputDataFrame.columns.values[:-1], InputDataFrame.columns.values[1:]):
            if Cols2 - Cols1 == 1:
                for XCoords, YCoords in zip(range(len(InputDataFrame[Cols1]) - 1), range(len(InputDataFrame[Cols2]) - 1)):
                    Function = np.sqrt(((float(InputDataFrame[Cols1].iloc[XCoords + 1]) - float(InputDataFrame[Cols1].iloc[XCoords]))**2) + ((float(InputDataFrame[Cols2].iloc[YCoords + 1]) - float(InputDataFrame[Cols2].iloc[YCoords]))**2))
                    DistanceVectors[Counter].append(Function)
                Counter += 1
            else:
                pass
        DataStructure = {
            BodyParts[Rng]:DistanceVectors[Rng] for Rng in range(len(DistanceVectors)) 
            }
        EuclideanDistanceDF = pd.DataFrame(DataStructure)
        return(EuclideanDistanceDF)

    def VarLoads(self, Init):
        DLCFrame = self.InheritImport(importFxn, ImportSource=Init.Source)
        DLCFrames = [self.preprocessFrame(InputDataFrame=Frames) for Frames in DLCFrame]
        PValCorrectedFrames = [self.checkPvals(InputDataFrame = Frames[0], CutOff = Init.CutOff) for Frames in DLCFrames]
        EuclideanDistanceFrames = [self.computeEuclidean(InputDataFrame = PValCorrectedFrames[Rng], BodyParts = DLCFrames[Rng][1])
                                   for Rng in range(len(PValCorrectedFrames))]
        BodyPartLists = [Lists[1] for Lists in DLCFrames]
        for Rng in range(len(EuclideanDistanceFrames)):
            self.InheritExport(EuclideanDistanceFrames[Rng], Init.ExportSource, "ED_Frame_{0}.csv".format(Rng))
        return(EuclideanDistanceFrames, BodyPartLists)

class hourlySum(initVars, Export):
    def InheritExport(self, ExportFile, ExportSource, FileName):
        return(FileExport().ExportFunction(Frame=ExportFile, Source=ExportSource, Name=FileName))
    
    def hourlySumFunction(self, IndFrames, BodyParts):
        sumVectors = [[] for _ in range(len(IndFrames.columns.values))]
        for Col, rng in zip(IndFrames.columns.values, range(len(IndFrames.columns.values))):
            SumFunction = sum(IndFrames[Col])
            sumVectors[rng].append(SumFunction)
        DataStructure = {
            BodyParts[Rng]:sumVectors[Rng] for Rng in range(len(sumVectors))
            }
        HourlySumFrame = pd.DataFrame(DataStructure)
        HourlySumFrame.index.name = "Hour"
        return(HourlySumFrame)
            
    def VarLoads(self, Init, EuclideanDistanceFrame, BodyParts):
        HourlySumFrames = [self.hourlySumFunction(IndFrames=EuclideanDistanceFrame[Rng], BodyParts=BodyParts[Rng])
                           for Rng in range(len(EuclideanDistanceFrame))]
        for Rng in range(len(HourlySumFrames)):
            self.InheritExport(HourlySumFrames[Rng], Init.ExportSource, "HourlySumFrame_{0}.csv".format(Rng))
        return(HourlySumFrames)
  
class mathFunctions(initVars, Export):
    def InheritExport(self, ExportFile, ExportSource, FileName):
        return(FileExport().ExportFunction(Frame=ExportFile, Source=ExportSource, Name=FileName))
    
    

      
if __name__ == '__main__':
    Init = InitializeVars(
        Path=r"F:\work\20191205-20200507T192029Z-001\20191205\162658_480x360DeepCut_resnet50_RatNov29shuffle1_1030000.csv",
        BodyPartList=["nose", "head", "body", "tail"], 
        PValCutOff=0.95, ExportSource=r"F:\work\20191205-20200507T192029Z-001\20191205\ED_Files"
        )

    importFxn = FileImport()
    # importFxn.ImportFunction_IfFile(Source=r"F:\work\20191205-20200507T192029Z-001\20191205\162658_480x360DeepCut_resnet50_RatNov29shuffle1_1030000.csv")
    EuclideanDistanceFrames = ComputeEuclideanDistance().VarLoads(Init)
    
    hourlySum().VarLoads(Init, EuclideanDistanceFrame=EuclideanDistanceFrames[0], BodyParts=EuclideanDistanceFrames[1])
    
