# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:40:38 2020

@author: Desktop
@Version: 2.0
"""
import pandas as pd
import numpy as np
import os

class Euclidean_Distance():
    source = []
    body_part_names = []
    cut_off = 0.95
    def len_bparts(self):
        return len(self.body_part_names)
    def conv_float(self, number):
        try:
            element = float(number)
            return element
        except ValueError:
            pass
    def split_data(self, vector, number_of_body_parts = 8):
        vector_length = len(vector)
        split = []
        for body_parts in range(number_of_body_parts):
           split1 = vector[body_parts*vector_length//number_of_body_parts:(body_parts+1)*vector_length//number_of_body_parts]
           split.append(split1)
        return split
    def file_locations(self):
        source = self.source
        if len(source) >= 1:
            files = [[] for _ in range(len(source))]
            i = 0
            while i < len(source):
                for filename in os.listdir(source[i]):
                    if filename.endswith(".csv"):
                        files[i].append(filename)
                    else:
                        pass
                i += 1
        else:
            print("Error: no source files found.")
        return files
    def export_location(self, export, name = '', sep = ',', index = False):
        ex = export.to_csv(name, sep = sep, index = index)
        return ex  
    def error_checking(self, probability_vector, coordinate_vector):
        for probs in range(len(probability_vector)):
            if probability_vector[probs] < (float(self.cut_off)):
                for x, y in enumerate(range(len(coordinate_vector))):
                    if y == probs:
                        if probs == 0:
                            coordinate_vector[x] = coordinate_vector[x]
                        else:
                            coordinate_vector[x] = coordinate_vector[probs-1]
        return coordinate_vector    
    def Euclidean_distance(self, coordinates):
        map_fxn_1 = map(
            lambda x, y: np.square(np.subtract(x, y)), coordinates[:-1], coordinates[1:] 
            )
        return list(map_fxn_1)
    def Euclidean_distance_2(self, x_cors, y_cors):
        map_fxn_2 = map(
            lambda x, y: np.sqrt((x + y)), x_cors, y_cors
            )
        return list(map_fxn_2)
    def dataframe_generator(self, filename):
        x_coordinates = []
        y_coordinates = []
        probabilities = []
        load = pd.read_csv(str(filename))
        df = pd.DataFrame(load)
        scorer_loc = df.iloc[:, 0].name
        df = df.drop(columns=scorer_loc, axis=1)
        y_col = 1
        L_col = 2
        for j, i in zip(range(len(df.columns)), df.columns):
            if (j % 3 == 0):
                for x in df[str(i)]:
                    x_cor = self.conv_float(x)
                    if (str(x_cor) != "None"):
                        x_coordinates.append(x_cor)
                    else:
                        pass
            elif ((j % 3 != 0) and (j == y_col)):
                for y in df[str(i)]:
                    y_cor = self.conv_float(y)
                    if (str(y_cor) != "None"):
                        y_coordinates.append(y_cor)
                    else:
                        pass
                y_col += 3
            elif ((j % 3 != 0) and (j == L_col)):
                for likelihood in df[str(i)]:
                    cutoff = self.conv_float(likelihood)
                    if (str(cutoff) != "None"):
                        probabilities.append(cutoff)
                    else:
                        pass
                L_col += 3
        ############
        #Split data
        ############
        Split_x = self.split_data(x_coordinates, self.len_bparts())
        Split_y = self.split_data(y_coordinates, self.len_bparts())
        Split_likelihood = self.split_data(probabilities, self.len_bparts())
        return Split_x, Split_y, Split_likelihood
    def Compute_Euclidean(self):
        files = self.file_locations()
        if len(self.source) == 1:        
            for files1, sources in zip(files, range(len(self.source))):
                os.chdir(str(self.source[sources]))
                for i_files in files1:
                    Master_lis = []
                    print("File {0} loaded".format(i_files))
                    for b_parts in range(0, int(self.len_bparts())):
                        print("processing {0}".format(self.body_part_names[b_parts]))
                        ############
                        #Error Checking
                        ############
                        Error_check_x_cor = self.error_checking(self.dataframe_generator(i_files)[2][b_parts], self.dataframe_generator(i_files)[0][b_parts])
                        Error_check_y_cor = self.error_checking(self.dataframe_generator(i_files)[2][b_parts], self.dataframe_generator(i_files)[1][b_parts])
                        print("Check finished")
                        ############
                        #ED computation
                        ############
                        comp_1_x = self.Euclidean_distance(Error_check_x_cor)
                        comp_1_y = self.Euclidean_distance(Error_check_y_cor)
                        
                        comp_2 = self.Euclidean_distance_2(comp_1_x, comp_1_y)
                        print("computation finished")
                        Master_lis.append(comp_2)
                    ##########
                    #Export
                    ##########
                    data_array = np.array([i for i in Master_lis])
                    Master_lis_df = pd.DataFrame([i for i in data_array])
                    Master_lis_df = Master_lis_df.transpose()
                    for b_parts in range(0, self.len_bparts()):
                        Master_lis_df = Master_lis_df.rename(columns={b_parts:"{0}".format(self.body_part_names[b_parts])})
                    self.export_location(Master_lis_df, name = './ED_Files/Complete_{0}'.format(i_files), sep = ',', index = False)
                    print("export complete!")
        elif len(self.source) > 1:
            for files2, sources in zip(files, range(len(self.source))):
                os.chdir(str(self.source[sources]))
                for i_files in files2:
                    Master_lis = []
                    print("File {0} loaded, from source {1}".format(i_files, sources))
                    for b_parts in range(0, int(self.len_bparts())):
                        print("processing {0}".format(self.body_part_names[b_parts]))
                        ############
                        #Error Checking
                        ############
                        Error_check_x_cor = self.error_checking(self.dataframe_generator(i_files)[2][b_parts], self.dataframe_generator(i_files)[0][b_parts])
                        Error_check_y_cor = self.error_checking(self.dataframe_generator(i_files)[2][b_parts], self.dataframe_generator(i_files)[1][b_parts])
                        print("Check finished")
                        ############
                        #ED computation
                        ############
                        comp_1_x = self.Euclidean_distance(Error_check_x_cor)
                        comp_1_y = self.Euclidean_distance(Error_check_y_cor)
                        
                        comp_2 = self.Euclidean_distance_2(comp_1_x, comp_1_y)
                        print("computation finished")
                        Master_lis.append(comp_2)
                    ##########
                    #Export
                    ##########
                    data_array = np.array([i for i in Master_lis])
                    Master_lis_df = pd.DataFrame([i for i in data_array])
                    Master_lis_df = Master_lis_df.transpose()
                    for b_parts in range(0, self.len_bparts()):
                        Master_lis_df = Master_lis_df.rename(columns={b_parts:"{0}".format(self.body_part_names[b_parts])})
                    self.export_location(Master_lis_df, name = './ED_Files/Complete_{0}'.format(i_files), sep = ',', index = False)
                    print("export complete!")
        End_statement = "All processing complete, check the export location(s)"
        return End_statement
##############
#Class calling
##############
ED=Euclidean_Distance()
ED.source = [r'C:\Users\Desktop\Desktop\work\20191205-20200507T192029Z-001\20191205\Test2']
ED.cut_off = 0.95
ED.body_part_names = ["Head", "Neck", "Back", "Tail", "back_right_leg","back_left_leg", "front_right_leg", "front_left_leg"]
print(ED.Compute_Euclidean())
