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
import scipy.integrate as integrate
import warnings

class InitializeVars:
    def __init__(self, Path, BodyPartList, PValCutOff, ExportSource, FramesPerSecond):
        self.Source = Path
        self.BodyParts = BodyPartList
        self.CutOff = PValCutOff
        self.ExportSource = ExportSource
        self.FPS = FramesPerSecond

class FileImport:
    def ImportFunction_IfPath(self, Source):
        CSVFileList = []
        for files in os.listdir(Source):
            if files.endswith(".csv"):
                CSVFileList.append(pd.read_csv("{0}/{1}".format(Source, files)))
        return(CSVFileList)

    def ImportFunction_IfFile(self, Source):
        CSVFileList = [pd.read_csv(Source)]
        return(CSVFileList)

class FileExport:
    """
    Export to same location as import
    """
    def ExportFunction(self, Frame, Source, Name):
        if os.path.isdir(Source) is True:
            ExportSource = "{0}/{1}".format(Source, Name)
            Frame.to_csv(ExportSource)
        elif os.path.isdir(Source) is False:
            warnings.warn("No valid export source was included, skipping export")

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
    
    def checkSpeedDistance(self, EuclideanDistanceFrame, FPS):
        OutlierFrames = [[] for _ in range(len(EuclideanDistanceFrame.columns.values))]
        VelocityDict = {
            EuclideanDistanceFrame.columns.values[rng]:[] for rng in range(len(EuclideanDistanceFrame.columns.values))
            }
        for Cols in EuclideanDistanceFrame:
            for VecDistance in EuclideanDistanceFrame[Cols]:
                Velocity = lambda Dist, Time: Dist/Time
                #Velocity in pixels per second
                ComputedVelocity = Velocity(Dist = float(VecDistance), Time = float(1/FPS))
                VelocityDict[Cols].append(ComputedVelocity)
        mp.plot(range(len(VelocityDict["head"])), VelocityDict["head"], linestyle='solid')        
        mp.show()
        print(VelocityDict)
        breakpoint()

    def createMovementPlot(self, InputDataFrame):
        mp.xlim(0, 480)
        mp.ylim(0, 360)
        Counter = 1
        while(Counter < max(InputDataFrame.columns.values)):
            mp.scatter(x=[float(vals) for vals in InputDataFrame[Counter]], y=[float(vals) for vals in InputDataFrame[Counter + 1]])
            Counter += 3
        mp.show()

    def VarLoads(self, Init):
        DLCFrame = self.InheritImport(importFxn, ImportSource=Init.Source)
        DLCFrames = [self.preprocessFrame(InputDataFrame=Frames) for Frames in DLCFrame]
        PValCorrectedFrames = [self.checkPvals(InputDataFrame = Frames[0], CutOff = Init.CutOff) for Frames in DLCFrames]
        EuclideanDistanceFrames = [self.computeEuclidean(InputDataFrame = PValCorrectedFrames[Rng], BodyParts = DLCFrames[Rng][1])
                                   for Rng in range(len(PValCorrectedFrames))]
        #self.createMovementPlot(InputDataFrame=PValCorrectedFrames[0])
        #return only the body part list of the first frame, should be the same as all others if dealing with path.
        #come up with a better way to do this though
        self.checkSpeedDistance(EuclideanDistanceFrame=EuclideanDistanceFrames[0], FPS = Init.FPS)
        BodyPartLists = DLCFrames[0][1]
        for Rng in range(len(EuclideanDistanceFrames)):
            self.InheritExport(EuclideanDistanceFrames[Rng], Init.ExportSource, "ED_Frame_{0}.csv".format(Rng))
        return(EuclideanDistanceFrames, BodyPartLists)

class hourlySum(initVars, Export):
    def InheritExport(self, ExportFile, ExportSource, FileName):
        return(FileExport().ExportFunction(Frame=ExportFile, Source=ExportSource, Name=FileName))

    def hourlySumFunction(self, IndFrames):
        sumVectors = []
        for Col, rng in zip(IndFrames.columns.values, range(len(IndFrames.columns.values))):
            SumFunction = sum(IndFrames[Col])
            sumVectors.append(SumFunction)
        return(sumVectors)
    
    def graphSums(self, InputFrame):
        for Cols in InputFrame.columns.values:
            mp.plot(InputFrame.index.values, InputFrame[Cols], label=Cols)
        mp.xlabel("Hour of Day")
        mp.ylabel("Total Motility per hour")
        mp.legend()
        mp.show()

    def VarLoads(self, Init, EuclideanDistanceFrame, BodyParts):
        HourlySumFrames = [self.hourlySumFunction(IndFrames=EuclideanDistanceFrame[Rng])
                           for Rng in range(len(EuclideanDistanceFrame))]
        HourlySumFrame = pd.DataFrame(HourlySumFrames, columns = BodyParts)
        self.graphSums(InputFrame=HourlySumFrame)
        for Rng in range(len(HourlySumFrames)):
            self.InheritExport(HourlySumFrames[Rng], Init.ExportSource, "HourlySumFrame_{0}.csv".format(Rng))
        return(HourlySumFrame)

class mathFunctions(initVars, Export):
    def InheritExport(self, ExportFile, ExportSource, FileName):
        return(FileExport().ExportFunction(Frame=ExportFile, Source=ExportSource, Name=FileName))
    
    def createLinearEquations(self, IndFrames):
        SlopeLists = [[] for _ in range(len(IndFrames.columns.values))]
        if len(IndFrames.index.values) > 1:
            for Cols, rng in zip(IndFrames, range(len(IndFrames))):
                for HSum, Index in zip(range(len(IndFrames[Cols]) - 1), range(len(IndFrames[Cols].index.values) - 1)):
                    Slope = (IndFrames[Cols][HSum + 1] - IndFrames[Cols][HSum])/(IndFrames[Cols].index.values[Index + 1] - IndFrames[Cols].index.values[Index])
                    Intercept = IndFrames[Cols][HSum] - (Slope * IndFrames[Cols].index.values[Index])
                    SlopeLists[rng].append((Slope, Intercept))
        elif len(IndFrames.index.values) == 1:
            pass
        return(SlopeLists)
    
    def integrateLinearFunctions(self, LinearEquation):
        IntegralValues = [[] for _ in range(len(LinearEquation))]
        for BodyParts, rng in zip(LinearEquation, range(len(LinearEquation))):
            for SIvals in BodyParts:
                Function = lambda x, a, b: a*x + b
                Integral = integrate.quad(Function, 0, 1, args=(SIvals[0], SIvals[1]))
                IntegralValues[rng].append(abs(Integral[0]))
        return(IntegralValues)
    
    def visualizeArea(self, IntegralFrame):
        X_Axis = range(len(IntegralFrame))
        X_Axis = np.arange(len(X_Axis))
        fig, ax = mp.subplots()
        Width = -0.2
        for cols in IntegralFrame.columns.values:
            ax.bar(x=X_Axis+Width, height=IntegralFrame[cols], width=0.1, label=str(cols))
            Width += 0.1
        ax.legend()
        mp.show()
        
    def VarLoads(self, Init, InputHourlySumFrame):
        LinearEquations = self.createLinearEquations(IndFrames = InputHourlySumFrame)
        Integrals = self.integrateLinearFunctions(LinearEquation=LinearEquations)
        DataStructure = {
            InputHourlySumFrame.columns.values[rng]:Integrals[rng] for rng in range(len(Integrals))
            }
        IntegralFrame = pd.DataFrame(DataStructure)
        self.visualizeArea(IntegralFrame=IntegralFrame)

if __name__ == '__main__':
    Init = InitializeVars(
        Path=r"F:\work\20191205-20200507T192029Z-001\20191205",
        BodyPartList=["nose", "head", "body", "tail"],
        PValCutOff=0.95, ExportSource="", FramesPerSecond = 4
        )
    importFxn = FileImport()
    # importFxn.ImportFunction_IfFile(Source=r"F:\work\20191205-20200507T192029Z-001\20191205\162658_480x360DeepCut_resnet50_RatNov29shuffle1_1030000.csv")
    EuclideanDistanceFrames = ComputeEuclideanDistance().VarLoads(Init)
    HourlySumFrame = hourlySum().VarLoads(Init, EuclideanDistanceFrame=EuclideanDistanceFrames[0], BodyParts=EuclideanDistanceFrames[1])
    mathFunctions().VarLoads(Init, InputHourlySumFrame=HourlySumFrame)
