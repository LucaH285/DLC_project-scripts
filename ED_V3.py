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
import math

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
        InputDataFrame=InputDataFrame.reset_index(drop = True)
        for Cols in InputDataFrame.columns.values:
            if Cols % 3 == 0:
                for Vals in InputDataFrame.index.values:
                    if float(InputDataFrame[Cols][Vals]) < CutOff:
                        #Using Iloc/loc here like that will take the whole row, all body parts.
                        #You can use the loc[row, column] to specify the columns, this should work
                        XCoordinates = Cols - 3
                        PValScore = Cols
                        if (Vals != 0):
                            PreviousRow = InputDataFrame.loc[Vals - 1, XCoordinates:PValScore]
                            InputDataFrame.loc[Vals, XCoordinates:PValScore] = PreviousRow
                        elif (Vals == 0):
                            #Some logic to control for the first index value containing a p-val < 0.95
                            pass
        return(InputDataFrame)
    
    def createSkeleton(self, InputDataFrame, BodyParts):
        print(InputDataFrame[[1]], InputDataFrame[[2]])
        OutlierFrames = []
        #Create vectors between body parts
        #Here we're dealing with 4 labels
        #################
        #Lambda Functions
        #################
        CoordVector = lambda X1, X2, Y1, Y2: [X2 - X1, Y2 - Y1]
        Norm = lambda Vec: np.sqrt(sum(x ** 2 for x in Vec))
        Normalize = lambda X, Mu, Sig: ((X - Mu)/Sig)
        LogScale = lambda X: np.log10(X)
        CreateVector = lambda Coord1, Coord2: [(float(y) - float(x)) for x, y in zip(Coord1, Coord2)]
        ComputeVectorAngle = lambda Vec1, Vec2: math.degrees(np.arccos(np.sum(float(i) * float(j) for i, j in zip(Vec1, Vec2))/(np.sqrt(np.sum(float(i)**2 for i in Vec1)) * (np.sqrt(np.sum(float(j)**2 for j in Vec2))))))
        # VectorAngles = lambda x, y: np.arccos((i * j for i, j in zip(x, y))/())
        #################
        
        InputDataFrame = InputDataFrame.drop([Cols for Cols in InputDataFrame.columns.values if Cols % 3 == 0], axis=1)
        CoordDict = {
            BodyParts[Ind]:[] for Ind in range(len(BodyParts))
            }
        DictPointer = 0
        for Cols in range(len(InputDataFrame.columns.values) - 1):
            if ((InputDataFrame.columns.values[Cols + 1]) - InputDataFrame.columns.values[Cols]) == 1:
                FrameCol1 = InputDataFrame.columns.values[Cols]
                FrameCol2 = InputDataFrame.columns.values[Cols + 1]
                CoordDict[BodyParts[DictPointer]] = [(x, y) for x, y in zip(InputDataFrame[FrameCol1], InputDataFrame[FrameCol2])]
                DictPointer += 1
        ComputeVectors = list(map(CreateVector, CoordDict["Head"], CoordDict["Body"]))
        ComputeVectors2 = list(map(CreateVector, CoordDict["Body"], CoordDict["Tail"]))
        ComputeAngles = list(map(ComputeVectorAngle, ComputeVectors, ComputeVectors2))
        ComputeNorm = list(map(Norm, ComputeVectors))
        ComputeNorm2 = list(map(Norm, ComputeVectors2))
        LogNorm = list(map(LogScale, ComputeNorm))
        LogNorm2 = list(map(LogScale, ComputeNorm2))
        print(ComputeNorm[0:3])
        print(LogNorm[0:3])
        mp.hist(LogNorm, bins=25, color = "Blue", label="Head-Body Vector")
        mp.hist(LogNorm2, bins=25, color="green", label="Body-Tail Vector")
        mp.xlabel("log-scaled Euclidean distance")
        mp.ylabel("Frequency in frames")
        mp.title("log-scaled ED vs. Frequency distribution")
        mp.legend()
        mp.show()
        mp.hist(ComputeAngles, bins=25, color="red")
        mp.xlabel("Angle (degrees)")
        mp.ylabel("Frequency in frames")
        mp.title("Head-Body and Body-Tail vector angle vs. Frequency of occurance")
        mp.show()
        InputDataFrame = InputDataFrame.rename(columns={InputDataFrame.columns.values[i]:i for i in range(len(InputDataFrame.columns.values))}).apply(pd.to_numeric)
        Counter = 0
        while(Counter < len(InputDataFrame.columns.values) - 3):
            SomeMap = list(map(CoordVector, InputDataFrame[Counter], InputDataFrame[Counter + 1], 
                               InputDataFrame[Counter + 2], InputDataFrame[Counter + 3]))
            SomeMap2 = list(map(Norm, SomeMap))
            Normal = list(map(Normalize, SomeMap2, [np.mean(SomeMap2)]*len(SomeMap2), [np.std(SomeMap2)]*len(SomeMap2)))
            LogScaled = [list(map(LogScale, SomeMap2))]
            mp.hist(x=LogScaled, bins=25)
            mp.xlabel("log scaled inter-label Euclidean distance")
            mp.ylabel("Number of frames")
            #mp.title("Norm from {0} to {1}".format())
            mp.show()
            Counter += 2
        # TTest = lambda 
        # for LabelDistances in OutlierFrames:
        breakpoint()
            
            

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
    
    def checkSpeedDistance(self, EuclideanDistanceFrame, FPS, BodyPart):
        OutlierFrames = [[] for _ in range(len(EuclideanDistanceFrame.columns.values))]
        VelocityDict = {
            EuclideanDistanceFrame.columns.values[rng]:[[], []] for rng in range(len(EuclideanDistanceFrame.columns.values))
            }
        for Cols in EuclideanDistanceFrame:
            Time = 1/FPS
            for VecDistance in EuclideanDistanceFrame[Cols]:
                Velocity = lambda Dist, Time: Dist/Time
                #Velocity in pixels per second
                ComputedVelocity = Velocity(Dist = float(VecDistance), Time = float(1/FPS))
                VelocityDict[Cols][0].append(Time)
                VelocityDict[Cols][1].append(ComputedVelocity)
                Time += 1/FPS
        AccelerationDict = {
            EuclideanDistanceFrame.columns.values[rng]:[[], []] for rng in range(len(EuclideanDistanceFrame.columns.values))
            }      
        for Keys in VelocityDict:
            Time2 = 1/FPS
            for Time, Velocity in zip(range(len(VelocityDict[Keys][0]) - 1), range(len(VelocityDict[Keys][1]) - 1)):
                Acceleration = lambda Vo, Vf, To, Tf: ((Vf - Vo)/(Tf - To))
                ComputedAcceleration = Acceleration(Vo=VelocityDict[Keys][1][Velocity], Vf = VelocityDict[Keys][1][Velocity + 1],
                                                    To = VelocityDict[Keys][0][Time], Tf = VelocityDict[Keys][0][Time + 1])
                AccelerationDict[Keys][0].append(Time2)
                AccelerationDict[Keys][1].append(ComputedAcceleration)
                Time2 += 1/FPS
        fig, ax = mp.subplots()
        ax.plot(AccelerationDict[BodyPart][0], AccelerationDict[BodyPart][1], linestyle='solid', label="Acceleration", color="red") 
        ax.plot(VelocityDict[BodyPart][0], VelocityDict[BodyPart][1], linestyle='solid', label = "Velocity")
        mp.xlabel("Elapsed Time, seconds")
        mp.ylabel("Acceleration (pixels/s^2) and Velocity (pixels/s)")
        ax.legend()
        mp.show()

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
        Skeleton = [self.createSkeleton(Frames, BodyParts = DLCFrames[0][1]) for Frames in PValCorrectedFrames]
        breakpoint()
        EuclideanDistanceFrames = [self.computeEuclidean(InputDataFrame = PValCorrectedFrames[Rng], BodyParts = DLCFrames[Rng][1])
                                   for Rng in range(len(PValCorrectedFrames))]
        
        #self.createMovementPlot(InputDataFrame=PValCorrectedFrames[0])
        print(DLCFrames[0][0])
        SampleUncorrected = self.computeEuclidean(InputDataFrame=DLCFrames[0][0], BodyParts=DLCFrames[0][1])
        print(SampleUncorrected)
        self.checkSpeedDistance(EuclideanDistanceFrame=SampleUncorrected, FPS = Init.FPS, BodyPart="Body")
        self.checkSpeedDistance(EuclideanDistanceFrame=EuclideanDistanceFrames[0], FPS = Init.FPS, BodyPart="Body")
        
        
        #return only the body part list of the first frame, should be the same as all others if dealing with path.
        #come up with a better way to do this though
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
        if len(HourlySumFrame.index.values) > 1:
            self.graphSums(InputFrame=HourlySumFrame)
        else:
            warnings.warn("Only a single .csv file was inputted for processing, cannot make a movement by hour graph!")
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
        Path=r"F:\work\TestVideos_NewNetwork\20191206-20200507T194022Z-001\20191206\RawVids\RawVideos2",
        BodyPartList=["nose", "head", "body", "tail"],
        PValCutOff=0.95, ExportSource="", FramesPerSecond = 4
        )
    importFxn = FileImport()
    # importFxn.ImportFunction_IfFile(Source=r"F:\work\20191205-20200507T192029Z-001\20191205\162658_480x360DeepCut_resnet50_RatNov29shuffle1_1030000.csv")
    EuclideanDistanceFrames = ComputeEuclideanDistance().VarLoads(Init)
    HourlySumFrame = hourlySum().VarLoads(Init, EuclideanDistanceFrame=EuclideanDistanceFrames[0], BodyParts=EuclideanDistanceFrames[1])
    print(HourlySumFrame)
    if len(HourlySumFrame.index.values) > 1:
        mathFunctions().VarLoads(Init, InputHourlySumFrame=HourlySumFrame)
    else:
        warnings.warn("Only a single .csv file was inputted for processing, cannot compute integrals or linearity!")
