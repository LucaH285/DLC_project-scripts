# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:47:06 2020

@author: Luca Hategan
@version: 1.0

A script designed to graph and process the Euclidean distance data produced from the ED_V2 script
"""
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
import os
import glob
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from scipy.integrate import quad

###########
#Body Parts
###########
Body_Part1 = 'Head'
Body_Part2 = 'Head'

###########
#Source Folder
###########
#Source_Folder_1 = os.chdir(r'C:\Users\Desktop\Desktop\work\20191207-20200429T225825Z-001\20191207\ED_Files')


###########
#Functions
###########
def load_files(source):
    extension = "csv"
    files = [files for files in glob.glob('*.{}'.format(extension))]
    return files

def conv_float(element):
    try:
       element = int(element)
       return element
    except ValueError:
        pass

def sumacross(vector):
    Total = 0
    for i in range(0, len(vector)):
        Total += vector[i]
    return Total

#Tail, Body, Head, Nose
def load_data(filename, Body_Part='Head'):
    Output_list = []
    Body_parts = ['Nose', 'Head', 'Body', 'Tail']
    load = pd.read_csv(filename)
    df = pd.DataFrame(load, columns = [i for i in Body_parts])
    
    for obj in df[Body_Part]:
        Output_list.append(obj)
    return Output_list


def extract(files, Body_Part=Body_Part1):
    out_list = []
    for i in range(len(files)):
        file = str(files[i])
        load = load_data(file, Body_Part=Body_Part)
        work = sumacross(load)
        out_list.append(work)
    return out_list

def averager(dataframe):
    avg_list = []
    for columns in dataframe:
        avg = dataframe.iloc[:, [columns]].mean(axis = 0)
        for nums in avg:
            avg_list.append(nums)
    return avg_list

def split_data(lis_t, split_by_hours = 24):
    list_size = len(lis_t)
    split = [lis_t[i*list_size//split_by_hours:(i+1)*list_size//split_by_hours]
             for i in range(split_by_hours)]
    return split

def Average(dataframe, columns_to_analyze = []):
    mean_list = []
    mean_column = dataframe.iloc[:, columns_to_analyze].mean(axis = 1)
    for i in mean_column:
        mean_list.append(i)
    return mean_list

def line_eqn(time_data, coordinate_data):
    slope = []
    intercepts = []
    x = 1
    y = 0 
    try:
        while x < len(time_data):
            for i in range(len(time_data)):
                if (time_data[i] == 23) and (time_data[(i+1)] == 0):
                    x_difference = ((24 - 23))
                    #print("corrected")
                else:
                    x_difference = ((time_data[x]) - time_data[y])
                    #print("completed")
                y_difference = ((coordinate_data[x] - coordinate_data[y]))
                slope_w = (y_difference)/(x_difference)
                slope.append(slope_w)
                x += 1
                y += 1
    except IndexError:
        pass
    n = 0
    while n < len(slope):
        for z in slope:
            work = ((-(time_data[n] * z)) + coordinate_data[n])
            intercepts.append(work)
            n += 1
    df = pd.DataFrame(data = {"slope":[z for z in slope],
                            "Intercept":[z for z in intercepts]})
    return df

def integrand(x, a, b):
    function = ((a*(x)) + b)
    return function

def integrator(slope_list, Intercept_list, bounds):
    integral_list = []
    sum_list = 0
    counter = 0
    for i, j in zip(slope_list, Intercept_list):
        if (bounds[counter] == 23) and (bounds[(counter+1)] == 0):
            I = quad(integrand, 23, 24, args = (i,j))
            integral_list.append(I[0])
            counter += 1
        else:
            I = quad(integrand, bounds[counter], bounds[(counter+1)], args=(i,j))
            integral_list.append(I[0])
            counter += 1
    for x in integral_list:
        sum_list += x
    return integral_list, sum_list


def seaborn_plot(dataset,times,limiter=80000):
    try:
        limiter == int(limiter)
        sns.set_style('white')
        sns.set_color_codes("muted")
        data_ = dataset
        time_list = []
        data_list = []
        red = 'r'
        blue = 'b'
        colors = []
        for i, v in enumerate(data_):
              if data_[i] > int(limiter):
                  data_list.append(('ED={:.2f}'.format(data_[i])))
                  time_list.append(("Hour {}:".format(times[i])))
                  color = red
                  colors.append(color)
              else:
                 color2 = blue
                 colors.append(color2)

        df = pd.DataFrame(data = {'times_new':time_list,
                                  'data_list':data_list})
        graph = sns.barplot(x=times, y=data_ , palette = colors)
        graph.set_xlabel("Time (Hours)", fontsize = 10, fontweight = 'bold');
        graph.set_ylabel("Euclidean Distance", fontsize = 10, fontweight = 'bold');
        graph.set_ylim([0, int(limiter)])
       
        handles = [bars for bars in graph.containers[0] if bars.get_height() > int(limiter)]
        labels = [f'Hour {" " if h < 10 else " "}{h}: ED={ed:,.0f}' for ed, h in zip(data_, times) if ed > int(limiter)]
        graph.legend(handles, labels, bbox_to_anchor=[1.02, 1], loc = 'upper left')
        
        if len(data_list) < 1:
            graph.get_legend().remove()
    
    except ValueError:
        sns.set_style('white')
        sns.set_color_codes("muted")
        data_ = dataset
        graph = sns.barplot(x=times, y=data_ , color='b')
        graph.set_xlabel("Time (Hours)", fontsize = 10, fontweight = 'bold');
        graph.set_ylabel("Euclidean Distance", fontsize = 10, fontweight = 'bold');
    return graph

def line_plot(Rat_Body1, timing, Body_label = str(Body_Part1), limiter = 220000, label_x = "Alpha"):
    sns.set(style = 'white')
    sns.set_style("ticks", {"xtick.major.size":8})
    sns.set_color_codes("muted")
    graph = sns.lineplot(x=timing, y=Rat_Body1, color='b', marker = 'o')
    if limiter != 'n':
        graph.set_ylim(0, int(limiter))
    else:
        pass
    graph.set_xlabel("Time (Hours)", fontsize = 10, fontweight = 'bold');
    graph.set_ylabel("Euclidean Distance", fontsize = 10, fontweight = 'bold');
    graph.set_xticklabels(rotation = 45, labels = ["spacer", "1pm", "6pm", "11pm", "4am", "9am"])
    
    labels = ["{}".format(str(label_x)), "Dark phase: 7pm-7am"]
    lines = [Line2D([0], [0], color = 'b', linewidth = 3, linestyle = '-'), 
    Line2D([0], [0], color = 'black', linewidth = 3, linestyle = 'dotted')]
     
    graph.axvline(6, color = 'black', zorder = 2.5, linestyle = 'dashed')
    graph.axvline(18, color = 'black', zorder = 2.5, linestyle = 'dashed')
    
    graph.legend(lines, labels, bbox_to_anchor=(1, 1.05))
    
    
    return graph

def alternate_plot(Rat_Body1, Rat_Body2, 
                   timing, Body_label1 = str(Body_Part1),
                   Body_label2 = str(Body_Part2), barWidth = 0.42, limiter = 80000):
    pos = np.arange(len(timing))
    pos2 = [x + barWidth for x in pos]
    
    g = mp.bar(pos, Rat_Body1, color = '#7f6d5f', width=barWidth, edgecolor='white', label=Body_label1)
    g = mp.bar(pos2, Rat_Body2, color = '#557f2d', width=barWidth, edgecolor='white', label=Body_label2)  
    g = mp.xlabel('Time (hours)', fontweight='bold')
    g = mp.ylabel("Euclidean Distance", fontweight='bold')
    # g = mp.ylim(80000)
    mp.legend()
    
    return mp.show(g)

def alternate_line_plot(Rat_Body1, Rat_Body2, timing, Body_label1 = str(Body_Part1),
                       Body_label2 = str(Body_Part2), limiter = 80000):
    try:
        limiter == int(limiter)
        
        figure, ax = mp.subplots()
        
        graph = ax.plot(timing, Rat_Body1, color = '#7f6d5f', label = Body_label1, marker = 'o', linestyle = 'dashed')
        graph = ax.plot(timing, Rat_Body2, color = '#557f2d', label = Body_label2, marker = 'o')
        graph = mp.xlabel("Time (Hours)", fontweight = 'bold')
        graph = mp.ylabel("Euclidean Distance", fontweight = 'bold')
        
        graph = ax.set_ylim(0, int(limiter))
        graph = mp.legend()
            
    except ValueError:
        graph = mp.plot(timing, Rat_Body1, color = '#7f6d5f', label = Body_label1, marker = 'o', linestyle = 'dashed')
        graph = mp.plot(timing, Rat_Body2, color = '#557f2d', label = Body_label2, marker = 'o')
        graph = mp.xlabel("Time (Hours)", fontweight = 'bold')
        graph = mp.ylabel("Euclidean Distance", fontweight = 'bold')
        
        graph = mp.legend()
        
    return graph
        
      
    
def alternate_sliced_line_plot(Rat_Body1, Rat_Body2, timing, Body_label1 =  str(Body_Part1),
                       Body_label2 =  str(Body_Part2), lower_bound1 = 55000, 
                       lower_bound2 = 70000, upper_bound = 95000):
 
    try:
        lower_bound1 == int(lower_bound1)
        lower_bound2 == int(lower_bound2)
        upper_bound == int(upper_bound)
        
        sns.set(style = 'white')
        sns.set_style("ticks", {"xtick.major.size":8})
        sns.set_color_codes("muted")
        
        figure, (ax2, ax1) = mp.subplots(2, 1, sharex = True)
        
        ax2.plot(timing, Rat_Body1, color = '#7f6d5f', marker = 'o', linestyle = 'dashed')
        ax2.plot(timing, Rat_Body2, color = '#557f2d', marker='o')
        
        ax1.plot(timing, Rat_Body1, color = '#7f6d5f', marker = 'o', linestyle = 'dashed')
        ax1.plot(timing, Rat_Body2, color = '#557f2d', marker='o')
        
        
        mp.xlabel("Time")
        mp.ylabel("ED")
        
        ax1.set_ylim(0, lower_bound1)
        ax2.set_ylim(lower_bound2, upper_bound)
        
        ax2.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        
        ax2.xaxis.tick_top()
        ax2.tick_params(labeltop = 'off')
        ax1.xaxis.tick_bottom()
        
        d = 0.025
        kwargs = dict(transform = ax2.transAxes, color = 'k', clip_on = False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        kwargs.update(transform = ax1.transAxes, color = 'k', clip_on = False)
        ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
        times_out = []
        data_out = []
        for i in range(len(Rat_Body1)):
            if Rat_Body1[i] > lower_bound1:
                times_out.append("Hour {}".format(timing[i]))
                data_out.append(" ED={:.2f}".format(Rat_Body1[i]))
        
        df = pd.DataFrame(data = {'times':times_out,
                                  'data':data_out})
        
        mp.legend(labels = df["times"] + df["data"], handles = None, bbox_to_anchor = [1.02, 1], loc = 'upper left')
        

        return figure
    
    
    except ValueError:
        g = mp.plot(timing, Rat_Body1, color = '#7f6d5f', label=Body_label1, marker = 'o', linestyle = 'dashed')
        g = mp.plot(timing, Rat_Body2, color = '#557f2d', label=Body_label2, marker='o')
        
        g = mp.xlabel('Time (hours)', fontweight = 'bold')
        g = mp.ylabel('Euclidean Distance', fontweight = 'bold')
        
        mp.legend()
        
        return g
    
    
def sliced_graph(times, dataset):            
    figure, (ax2, ax) = mp.subplots(2, 1, sharex=True)
    
    sns.set(style = 'white')
    sns.set_style("ticks", {"xtick.major.size":8})
    sns.set_color_codes("muted")
    
    ax.plot(times, dataset, color = 'b')
    ax2.plot(times, dataset, color = 'b')
    
    ax.set_ylim(0, 120000)
    ax2.set_ylim(190000, 260000)
    
    ax2.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax2.xaxis.tick_top()
    ax2.tick_params(labeltop = 'off')
    ax.xaxis.tick_bottom()
    
    d = 0.015
    kwargs = dict(transform = ax2.transAxes, color = 'k', clip_on = False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform = ax.transAxes, color = 'k', clip_on = False)
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    for i in range(len(dataset)):
        if dataset[i] >= 500000:
            label = '{:.2f}'.format(dataset[i])
            ax2.annotate(label, (times[i],dataset[i]), textcoords = 'offset points', ha = 'center', xytext = (0,0))
            ax.annotate(label, (times[i],dataset[i]), textcoords = 'offset points', ha = 'center', xytext = (0, 3))
            #ax2.spines[labelling].set_visible(True)
            
    return figure     

user = input("Create averager? [y/n]")
if user == 'y':
    count = input("days to analyze? [enter a number]")
    genotype = input("Genotypes to compare? [1/2]")
    if int(genotype) == 2:
        out = 1
        master_list = []
        master_list2 = []
        while out <= int(count):
            source_folder = input("Copy the source folder location for the WT animal__{}".format(out))
            file_load = load_files(source = os.chdir(str(source_folder)))
            work = extract(file_load, Body_Part=Body_Part1)
            master_list.append(work)
            out += 1  
        df = pd.DataFrame(master_list)
        average1 = averager(df)
        out2 = 1
        while out2 <= int(count):
            source_folder = input("Copy the source folder location for the KO animal__{}".format((out2)))
            file_load2 = load_files(source = os.chdir(str(source_folder)))
            work = extract(file_load2, Body_Part=Body_Part2)
            master_list2.append(work)
            out2 += 1
        df2 = pd.DataFrame(master_list2)
        average2 = averager(df2)
        timing = [i for i in range(0, 24)]
        # print(master_list2)
        # print(average2)
        plt_type = input('Line or bar plot? [line/bar]')
        
        out3 = 0
        while out3 != 1:
            if plt_type == 'line':
                line_plot1 = alternate_line_plot(average1, average2, timing, 
                                                Body_label1 = str(Body_Part1), Body_label2 = str(Body_Part2), limiter = 'n')
                out3 += 1
            elif plt_type == 'bar':
                bar_plot = alternate_plot(average1, average2, 
                           timing, Body_label1 = str(Body_Part1),
                           Body_label2 = str(Body_Part2), barWidth = 0.42)
                out3 += 1
    elif int(genotype) == 1:
        out = 1
        averager_list1 = []
        while out <= int(count):
            source_folder = input("Copy the source folder location for the animal, pass_{}".format(out))
            load = load_files(source = os.chdir(str(source_folder)))
            extraction = extract(load, Body_Part = Body_Part1)
            if len(extraction) == 24:
                averager_list1.append(extraction)
            elif len(extraction) < 24:
                start_time = input("Enter the start time [hour]")
                end_time = input("Enter the end time [hour]")
                temp_list = []
                for i in range(int(start_time), int(end_time)):
                    temp_list.append(np.nan)
                append_position = input("Append at the end or the beginning? [b/e]")
                if append_position == 'b':
                    extraction = temp_list + extraction
                    averager_list1.append(extraction)
                elif append_position == 'e':
                    extraction.extend(temp_list)
                    averager_list1.append(extraction)
            out += 1
        df_single_genotype = pd.DataFrame(averager_list1)
        average_single_Gt = averager(df_single_genotype)
        
        # print(df_single_genotype)
        # print(average_single_Gt)
        
        reorder = [13, 14, 15, 16, 17, 18 ,19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11, 12]
    
        
        Average_reorder = []
        hour = 0
        while hour < len(average_single_Gt):
            for i in reorder:
                Average_reorder.append(average_single_Gt[i])
                hour += 1
        # print(Average_reorder)
        timing = [i for i in range(0, 24)]
        #############
        #Label name
        #############
        label_x = "Genotype Sex: ID-number; recording period (ie: 20200111-14)"
        plot = line_plot(Average_reorder, timing, limiter = 120000, label_x = label_x)
        
        single_day_reorder = []
        df_single_genotype = df_single_genotype.transpose()
        for columns in df_single_genotype:
            for numbers in df_single_genotype[columns]:
                single_day_reorder.append(numbers)
        split = split_data(single_day_reorder, split_by_hours = int(count))
        temp_list = []
        days = 0
        while days < len(split):
            for i in reorder:
                temp_list.append(split[days][i])
            days += 1
        single_day_integrator = split_data(temp_list, split_by_hours = int(count))
        
        ###############
        #Daily per hour data of the individuls 
        ###############
        if int(count) <= 4:
            #df_single_genotype = df_single_genotype.transpose()
            hourly_list = []
            for columns in df_single_genotype:
                for numbers in df_single_genotype[columns]:
                    hourly_list.append(numbers) 
            split_by_day = split_data(hourly_list, split_by_hours = int(count))
            reordered_split_day = []
            days = 0
            while days < len(split_by_day):
                for hours in reorder:
                    reordered_split_day.append(split_by_day[days][hours])
                days += 1
            split_by_day_2 = split_data(reordered_split_day, split_by_hours = int(count))
            
            per_hour_df = pd.DataFrame(split_by_day_2)
            per_hour_df = per_hour_df.transpose()
            for col in range(0, int(count)):
                per_hour_df = per_hour_df.rename(columns={col:"Day {}".format((col+1))})
            
            index_list = ["1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm", "9pm",
                          "10pm", "11pm", "12am", "1am", "2am", "3am", "4am", "5am", "6am",
                          "7am", "8am", "9am", "10am", "11am", "12pm"]
            counter = 0
            while counter < len(index_list):
                for index in range(0, len(reorder)):
                    per_hour_df = per_hour_df.rename(index={index:"{}".format((index_list[counter]))})
                    counter += 1
            single_day_integration_darkphase = []
            single_day_integration_lightphase = []
            
            # print(per_hour_df)
            #############
            #Darkphase
            #############
            #for numbers in darkphase and not in light phase
            #for numbers2 in light phase and not in dark phase
            # print(single_day_integrator)
            for obj in range(len(single_day_integrator)):
                if any((str(number) == str("nan")) for number in single_day_integrator[obj][6:19]):
                    single_day_integration_darkphase.append(np.nan)
                elif ((any((str(number2) == str("nan") or (str(number3) == str("nan"))) for number2, number3 in zip(single_day_integrator[obj][0:7],
                                                                                                                  single_day_integrator[obj][18:24])) and (any(str(number4) != str("nan")) for number4 in single_day_integrator[obj][6:19]))):
                    temp_lis = []
                    for nums in single_day_integrator[obj][6:19]:
                        temp_lis.append(nums)
                    function1 = line_eqn(reorder[6:19], temp_lis)
                    Integrator1 = integrator(function1["slope"], function1["Intercept"], reorder[6:19])
                    single_day_integration_darkphase.append(Integrator1[1])
                else:
                    function1 = line_eqn(reorder[6:19], single_day_integrator[obj][6:19])
                    Integrator1 = integrator(function1["slope"], function1["Intercept"], reorder[6:19])
                    single_day_integration_darkphase.append(Integrator1[1])
                #iterate through this loop n times. 
                check2 = 0
                
                print(function1)
                print(Integrator1[0])

            #############
            #Lightphase
            #############
            #The for loops are integrating too many times.
            for obj in range(len(single_day_integrator)):
                if any((str(float_nums1) == str("nan")) or (str(float_nums2) == str("nan")) for float_nums1, float_nums2 in zip(single_day_integrator[obj][0:7], single_day_integrator[obj][18:24])):
                    single_day_integration_lightphase.append(np.nan)
                elif (any(str(nums) == (str("nan")) for nums in single_day_integrator[obj][6:19]) and (any((str(nums1) != str("nan")) or ((str(nums2) != str("nan")))) for nums1, nums2 in zip(single_day_integrator[obj][0:7], single_day_integrator[obj][18:24]))):
                    temp_lis_left = []
                    temp_lis_right = []
                    for numbers in single_day_integrator[obj][0:7]:
                        temp_lis_left.append(numbers)
                    for numbers in single_day_integrator[obj][18:24]:
                        temp_lis_right.append(numbers)
                    function2 = line_eqn(reorder[0:7], temp_lis_left)
                    Integrator2 = integrator(function2["slope"], function2["Intercept"], reorder[0:7])
                    #####
                    function3 = line_eqn(reorder[18:24], temp_lis_right)
                    Integrator3 = integrator(function3["slope"], function3["Intercept"], reorder[18:24])
                    Integrator_sum = (Integrator2[1] + Integrator3[1])
                    single_day_integration_lightphase.append(Integrator_sum)
                else:
                    function2 = line_eqn(reorder[0:7], single_day_integrator[obj][0:7])
                    Integrator2 = integrator(function2["slope"], function2["Intercept"], reorder[0:7])
                    #single_day_integration_lightphase.append(Integrator2[1])
                    ########
                    function3 = line_eqn(reorder[18:24], single_day_integrator[obj][18:24])
                    Integrator3 = integrator(function3["slope"], function3["Intercept"], reorder[18:24])
                    Integrator_sum = (Integrator2[1] + Integrator3[1])
                    single_day_integration_lightphase.append(Integrator_sum)
               
            print(single_day_integration_lightphase)
            single_day_df = pd.DataFrame(data={"Lightphase":single_day_integration_lightphase,
                                                "Darkphase":single_day_integration_darkphase,
                                                "Days":("Day 1")}) #, "Day 2", "Day 3", "Day 4"
            single_day_df = single_day_df.set_index(["Days"])
            
            print(single_day_df)
            
    
            single_day_df.to_csv("Individual_day_integrated_phase.csv")
            per_hour_df.to_csv("Per_hour_data.csv")
            
        elif int(count) > 4:
            pass
                
                
        #################
        #Integration for the Average
        #################           
        Dark_phase_vector = []
        Light_phase_vector = []
        
        #############
        #Darkphase integration
        #############
        function1 = line_eqn(reorder[6:19], Average_reorder[6:19])
        Integrator1 = integrator(function1["slope"], function1["Intercept"], reorder[6:19])
        Dark_phase_vector.append(Integrator1[1])
        
        #############
        #Lightphase integration
        #############
        function2 = line_eqn(reorder[0:7], Average_reorder[0:7])
        Integrator2 = integrator(function2["slope"], function2["Intercept"], reorder[0:7])
        function3 = line_eqn(reorder[18:24], Average_reorder[18:24])
        Integrator3 = integrator(function3["slope"], function3["Intercept"], reorder[18:24])
        Integrator_sum = (Integrator2[1] + Integrator3[1])
        Light_phase_vector.append(Integrator_sum)
        
        df = pd.DataFrame(data={"Darkphase Area":Dark_phase_vector,
                                "Lightphase Area":Light_phase_vector})
        df2 = pd.DataFrame(data={"Averaged Hourly ED, {0} days".format(count):Average_reorder,
                                 "Hours":["1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm", "9pm",
                                          "10pm", "11pm", "12am", "1am", "2am", "3am", "4am", "5am", "6am",
                                          "7am", "8am", "9am", "10am", "11am", "12pm"]})
        df2 = df2.set_index(["Hours"])
        
        print(df)
        print(df2)
        
        df.to_csv("Averaged_Integrated_Phase.csv")
        df2.to_csv("Averaged_Hourly_ED.csv")
              
        mp.show()        
        
       
else:
##########
#Function calling - main loop
##########
    files = load_files()
    Rat_Body_sum1 = extract(files, Body_Part = Body_Part1)
    
    user_dir_chng = input("Add another file? [y/n]")
    out = 0
    while out < 1:
        if user_dir_chng == 'y':
            Source = input("Copy and paste the source folder location")
            Source_Folder_2 = os.chdir(Source)
            files2 = load_files(source = Source_Folder_2)
            Rat_Body_sum2 = extract(files2, Body_Part = Body_Part2)
            out += 1
        else:
            Rat_Body_sum2 = []
            out += 1

    ##########
    #Output portion
    ##########
    timing = [i for i in range(0, 24)]
    user_in = input("Create a multiday lineplot (maximum of 5)? [y/n]")
    if user_in == 'n':
        if len(Rat_Body_sum2) > 0:
            out = 0
            while out != 1:
                user_input = input("Line plot, bar plot or side by side bar plots? [line/bar/side by side]")
                if user_input == 'bar':
                    plot = alternate_plot(Rat_Body_sum1, Rat_Body_sum2, 
                                          timing, Body_label1 = Body_Part1, Body_label2 = Body_Part2, barWidth=0.42)
                    out += 1
                elif user_input == 'line':
                    plot = alternate_line_plot(Rat_Body_sum1, Rat_Body_sum2, timing, Body_label1 = Body_Part1,
                                               Body_label2 = Body_Part2, limiter = 'n')
                    out += 1
                elif user_input == 'side by side':
                    plot = seaborn_plot(Rat_Body_sum1, timing)
                    plot2 = seaborn_plot(Rat_Body_sum2, timing)
                    
                    mp.show(plot)
                    mp.show(plot2)
                    out += 1
                else:
                    out == 0
                    print('invalid choice')
            df3 = pd.DataFrame(data={'times':timing,
                                      '{}'.format(Body_Part1):Rat_Body_sum1,
                                      '{}'.format(Body_Part2):Rat_Body_sum2})
            df3.to_csv('./sum2/{0}_sum_data.csv'.format(('1' + Body_Part1) + ('2' + Body_Part2)), sep = ',', index = False)
            
        else:
            if len(timing) != len(Rat_Body_sum1):
                string_list = []
                definition_list = []
                for i in files:
                    string_list.append(i.split('_'))
                for objects in string_list:
                    for numbers in objects:
                        conversion = conv_float(numbers)
                        if conversion != None:
                            definition_list.append(conversion)
                try:
                    graph = sns.barplot(x = definition_list, y = Rat_Body_sum1, color = 'b')
                    graph.set_xlabel("Times Imported", fontsize = "12")
                    graph.set_ylabel("Euclidean Distance")
                    graph.tick_params(labelsize = 6.5)
                    # plot = seaborn_plot(Rat_Body_sum1, definition_list)
                    mp.show(graph)
                    print("IMPORTANT!! This analysis was not conducted for the full 24 hour period. Numbers on x axis represent the hours recorded, correspond to numbers on input file")
                except ValueError:
                    pass
            else:
                Finished = 0
                while Finished < 1:
                    user_in = input("Bar graph or sliced bar graph? [bar/sliced]")
                    if user_in == 'bar':
                        limiter = input("Enter limiter value [number/n]")
                        plot = seaborn_plot(Rat_Body_sum1, timing, limiter = limiter)
                        mp.show(plot)  
                        Finished += 1
                    elif user_in == 'sliced':
                        plot = sliced_graph(timing, Rat_Body_sum1)
                        mp.show(plot) 
                        Finished += 1
    
    elif user_in == 'y':              
        Average_ = input("Average the data in each sex and genotype category? [y/n]")
        if Average_ == 'y':
            Male_WT_numbers = input("enter the number of WT Males")
            Male_KO_numbers = input("enter the number of KO Males")
            M_WT_list = []
            M_KO_list = []
            Male_WT_control = 0
            #Make WT
            WT_days = input("enter the number of days for the WT Male")
            KO_days = input("enter the number of days for the KO Male")
            while Male_WT_control < int(Male_WT_numbers):
                day_control = 0
                while day_control < int(WT_days):
                    file = input("enter the folder directory for Male {0}, WT day {1}, or skip [n]".format((Male_WT_control+1), (day_control+1)))
                    if file != 'n':
                        source_M_WT = os.chdir(file)
                        files_M_WT = load_files(source = source_M_WT)
                        extract_M_WT = extract(files_M_WT, Body_Part=Body_Part1)
                        if len(extract_M_WT) == 24:
                            M_WT_list.append(extract_M_WT)
                        elif len(extract_M_WT) < 24:
                            check = 0
                            while check < 1:
                                start_time = input("enter the start time")
                                end_time = input("enter the end time")
                                if (start_time != int(start_time)) or (end_time != int(end_time)):
                                    check == 0 #For some element in range(0, 24) | input == e
                                elif (start_time == int(start_time)) and (end_time == int(end_time)):
                                    pass
                                check += 1
                            blank_list = []
                            for i in range(int(start_time), int(end_time)):
                                blank_list.append(np.nan)
                            append_at_beginning = input("append at the beginning or end? [b/e]")
                            if append_at_beginning == 'b':
                                extract_M_WT = blank_list + extract_M_WT
                                M_WT_list.append(extract_M_WT)
                            elif append_at_beginning == 'e':
                                extract_M_WT.extend(blank_list)
                                M_WT_list.append(extract_M_WT)    
                    elif file == 'n':
                        intermediate_list1 = []
                        for i in range(0, 24):
                            intermediate_list1.append(np.nan)
                        M_WT_list.append(intermediate_list1)
                    day_control += 1
                Male_WT_control += 1  
            #Male KO
            Male_KO_control = 0
            while Male_KO_control < int(Male_KO_numbers):
                day_control_KO = 0
                while day_control_KO < int(KO_days):
                    file = input("enter the folder directory for Male {0}, KO day {1}, or skip [n]".format((Male_KO_control+1), (day_control_KO+1)))
                    if file != 'n':
                        source_M_KO = os.chdir(file)
                        files_M_KO = load_files(source = source_M_KO)
                        extract_M_KO = extract(files_M_KO, Body_Part=Body_Part1)
                        if len(extract_M_KO) == 24:
                            M_KO_list.append(extract_M_KO)
                        elif len(extract_M_KO) < 24:
                            start_time2 = input("enter the start time")
                            end_time2 = input("enter the end time")
                            blank_list = []
                            for i in range(int(start_time), int(end_time)):
                                blank_list.append(np.nan)
                            append_at_beginning2 = input("append at the beginning or end? [b/e]")
                            if append_at_beginning2 == 'b':
                                extract_M_KO = blank_list + extract_M_KO
                                M_KO_list.append(extract_M_KO)
                            elif append_at_beginning == 'e':
                                extract_M_KO.extend(blank_list)
                                M_KO_list.append(extract_M_KO)
                    elif file == 'n':
                        intermediate_list2 = []
                        for i in range(0, 24):
                            intermediate_list2.append(np.nan)
                        M_KO_list.append(intermediate_list2)
                    day_control_KO += 1
                Male_KO_control += 1
            #Female WT
            
            Female_WT_numbers = input("enter the number of WT Females")
            Female_KO_numbers = input("enter the number of KO Females")
            F_WT_list = []
            F_KO_list = []
            Female_WT_control = 0
            F_WT_days = input("enter the number of days for the WT Female")
            F_KO_days = input("enter the number of days for the KO Female")
            while Female_WT_control < int(Female_WT_numbers):
                day_control_F_WT = 0
                while day_control_F_WT < int(F_WT_days):
                    file = input("enter the folder directory for Female {0}, WT day {1}, or skip [n]".format((Female_WT_control+1), (day_control_F_WT+1)))
                    if file != 'n':
                        source_F_WT = os.chdir(file)
                        files_F_WT = load_files(source = source_F_WT)
                        extract_F_WT = extract(files_F_WT, Body_Part=Body_Part1)
                        if len(extract_F_WT) == 24:    
                            F_WT_list.append(extract_F_WT)
                        elif len(extract_F_WT) < 24:
                            start_time3 = input("enter the start time")
                            end_time3 = input("enter the end time")
                            blank_list3 = []
                            for i in range(int(start_time3), int(end_time3)):
                                blank_list3.append(np.nan)
                            append_from_beginning = input("append from beginning or end? [b/e]")
                            if append_from_beginning == 'b':
                                extract_F_WT = blank_list3 + extract_F_WT
                                F_WT_list.append(extract_F_WT)
                            elif append_from_beginning == 'e':
                                extract_F_WT.extend(blank_list3)
                                F_WT_list.append(extract_F_WT)
                    elif file == 'n':
                        intermediate_list3 = []
                        for i in range(0, 24):
                            intermediate_list3.append(np.nan)
                        F_WT_list.append(intermediate_list3)
                    day_control_F_WT += 1
                Female_WT_control += 1
            #Female KO
            Female_KO_control = 0
            while Female_KO_control < int(Female_KO_numbers):
                day_control_F_KO = 0
                while day_control_F_KO < int(F_KO_days):
                    file = input("enter the folder directory for Female {0}, KO day {1}, or skip [n]".format((Female_KO_control+1), (day_control_F_KO+1)))
                    if file != 'n':
                        source_F_KO = os.chdir(file)
                        files_F_KO = load_files(source = source_F_KO)
                        extract_F_KO = extract(files_F_KO, Body_Part=Body_Part1)
                        if len(extract_F_KO) == 24:
                            F_KO_list.append(extract_F_KO)
                        elif len(extract_F_KO) < 24:
                            start_time4 = input("enter the start time")
                            end_time4 = input("enter the end time")
                            blank_list4 = []
                            for i in range(int(start_time4), int(end_time4)):
                                blank_list4.append(np.nan)
                            append_from_beginning2 = input("append from the beginning or end? [b/e]")
                            if append_from_beginning2 == 'b':
                                extract_F_KO = blank_list4 + extract_F_KO
                                F_KO_list.append(extract_F_KO)
                            elif append_from_beginning2 == 'e':
                                extract_F_KO.extend(blank_list)
                                F_KO_list.append(extract_F_KO)
                    elif file == 'n':
                        intermediate_list4 = []
                        for i in range(0, 24):
                            intermediate_list4.append(np.nan)
                        F_KO_list.append(intermediate_list4)
                    day_control_F_KO += 1
                Female_KO_control += 1
                 
            Data_Frame_M_WT = pd.DataFrame(M_WT_list)
            Data_Frame_M_WT = Data_Frame_M_WT.transpose()
            Data_Frame_M_KO = pd.DataFrame(M_KO_list)
            Data_Frame_M_KO = Data_Frame_M_KO.transpose()
            Data_Frame_F_WT = pd.DataFrame(F_WT_list)
            Data_Frame_F_WT = Data_Frame_F_WT.transpose()
            Data_Frame_F_KO = pd.DataFrame(F_KO_list)
            Data_Frame_F_KO = Data_Frame_F_KO.transpose()
                
            #print(Data_Frame_F_WT)
            class average():
                data_in = []
                first_day = 0
                final_day = 2
                #If genotype > 2
                multi_genotype_vector = []
                def Average(self):
                    x = self.first_day
                    j = self.final_day
                    mean_list = []
                    while (x < (self.final_day)) and (j < (2*(self.final_day))):
                        mean_column = self.data_in.iloc[:, [x, j]].mean(axis = 1)
                        for i in mean_column:
                            mean_list.append(i)
                        x += 1
                        j += 1
                        print(j)
                    return mean_list
                
                def multi_genotype_average(self):
                    y=self.final_day
                    x=self.first_day
                    genotype_vector = self.multi_genotype_vector
                    mean_list_multi_genotype = []
                    while (x < (self.final_day)) and (y < (2*(self.final_day))):
                        mean_column_multi_genotype = self.data_in.iloc[:, genotype_vector].mean(axis = 1)
                        for i in mean_column_multi_genotype:
                            mean_list_multi_genotype.append(i)
                        for i in range(len(genotype_vector)):
                            genotype_vector[i] += 1
                        x += 1
                        y += 1
                    return mean_list_multi_genotype
                        
            ## Review this tomorrow
            split_by = 2
            
            M_WT = average()
            M_WT.data_in = Data_Frame_M_WT
            M_WT.final_day = split_by
            M_WT.first_day = 0
            M_WT_avg = M_WT.Average()
            print(M_WT_avg)
            
            M_KO = average()
            M_KO.data_in = Data_Frame_M_KO
            M_KO.final_day = split_by
            M_KO.first_day = 0
            M_KO_avg = M_KO.Average()            
        
            F_WT = average()
            F_WT.data_in = Data_Frame_F_WT
            F_WT.first_day = 0
            F_WT.final_day = split_by
            F_WT_avg = F_WT.Average()
           
            F_KO = average()
            F_KO.data_in = Data_Frame_F_KO
            F_KO.first_day = 0
            F_KO.final_day = split_by
            F_KO_avg = F_KO.Average()
            
            
            df = pd.DataFrame(data={"WT_Male":M_WT_avg, "M_KO":M_KO_avg,
                                    "F_WT":F_WT_avg, "F_KO":F_KO_avg})
            
            M_WT_split = split_data(M_WT_avg, split_by_hours=split_by)
            M_KO_split = split_data(M_KO_avg, split_by_hours=split_by)
            F_WT_split = split_data(F_WT_avg, split_by_hours=split_by)
            F_KO_split = split_data(F_KO_avg, split_by_hours=split_by)
            
            reorder = [13, 14, 15, 16, 17, 18 ,19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12]
            M_WT_reorder = []
            M_KO_reorder = []
            F_WT_reorder = []
            F_KO_reorder = []
            
            day_F = 0
            while day_F < split_by:
                for i in reorder:
                    if len(F_WT_split[day_F]) != 0:
                        F_WT_reorder.append(F_WT_split[day_F][i])
                    else:
                        pass
                    if len(F_KO_split[day_F]) != 0:
                        F_KO_reorder.append(F_KO_split[day_F][i])
                    else:
                        pass
                day_F += 1
            day_M = 0
            while day_M < split_by:
                for i in reorder:
                    if len(M_WT_split[day_M]) != 0:
                        M_WT_reorder.append(M_WT_split[day_M][i])
                    else:
                        pass
                    if len(M_KO_split[day_M]) != 0: 
                        M_KO_reorder.append(M_KO_split[day_M][i])
                    else:
                        pass
                day_M += 1
                
            M_WT_reorder = split_data(M_WT_reorder, split_by_hours=split_by)
            M_KO_reorder = split_data(M_KO_reorder, split_by_hours=split_by)
            F_WT_reorder = split_data(F_WT_reorder, split_by_hours=split_by)
            F_KO_reorder = split_data(F_KO_reorder, split_by_hours=split_by)
            
            
            total_df_M = pd.DataFrame(data = {"Male, WT":M_WT_reorder,
                                                  "Male, KO":M_KO_reorder})
            total_df_F = pd.DataFrame(data = {"Female, WT":F_WT_reorder,
                                              "Female, KO":F_KO_reorder})
    
            total_df_invert_M = total_df_M.transpose()
            total_df_invert_F = total_df_F.transpose()
            
            for i in range(0, 2):
                total_df_invert_M = total_df_invert_M.rename(columns={i:("Day {}".format((i+1)))})
            for i in range(0, 2):
                total_df_invert_F = total_df_invert_F.rename(columns={i:("Day {}".format((i+1)))})
                    
        elif Average_ == 'n':
                user_in_M = input("enter the number of days for the Males [number]")
                number_of_days_Male = int(user_in_M)
                set_day = 0
                M_WT_list = []
                M_KO_list = []
                while set_day < int(number_of_days_Male):
                    user_in_M = input("Enter the folder directory for Male WT day {0}, or skip [n]; pass_{1}".format((set_day + 1), set_day))
                    if user_in_M != 'n':
                        source_M = os.chdir(user_in_M)
                        files_M = load_files(source = source_M)
                        extract_M_WT = extract(files_M, Body_Part = Body_Part1)
                        M_WT_list.append(extract_M_WT)
                    elif user_in_M == 'n':
                        fill1 = [0]
                        intermediate_list1 = []
                        for i in range(0, 24):
                            intermediate_list1.append(fill1[0])
                        M_WT_list.append(intermediate_list1)
                    else:
                        pass
                    user_in_M = input("Enter the folder directory for Male KO day {0}, or skip [n]; pass_{1}".format((set_day + 1), set_day))
                    if user_in_M != 'n':
                        source_M = os.chdir(user_in_M)
                        files_M = load_files(source = source_M)
                        extract_M_KO = extract(files_M, Body_Part = Body_Part1)
                        M_KO_list.append(extract_M_KO)
                    elif user_in_M == 'n':
                        fill2 = [0]
                        intermediate_list2 = []
                        for i in range(0, 24):
                            intermediate_list2.append(fill2[0])
                        M_KO_list.append(intermediate_list2)
                    else:
                        pass
                    set_day += 1
                
                user_in_F = input("enter the number of days for the Females [number]")
                number_of_days_Female = int(user_in_F)
                
                set_day = 0
                F_KO_list = []
                F_WT_list = []
                while set_day < int(number_of_days_Female):
                    user_in_F = input("Enter the folder directory for Female WT day {0}, or skip [n]; pass_{1}".format((set_day + 1), set_day))
                    if user_in_F != 'n':
                        source_F = os.chdir(user_in_F)
                        files_F = load_files(source = source_F)
                        extract_F_WT = extract(files_F, Body_Part = Body_Part1)
                        F_WT_list.append(extract_F_WT)
                    elif user_in_F == 'n':
                        fill3 = [0]
                        intermediate_list3 = []
                        for i in range(0, 24):
                            intermediate_list3.append(fill3[0])
                        F_WT_list.append(intermediate_list3)
                    else:
                        pass
                    user_in_F = input("Enter the folder directory for Female KO day {0}, or skip [n]; pass_{1}".format((set_day + 1), set_day))
                    if user_in_F != 'n':
                        source_F = os.chdir(user_in_F)
                        files_F = load_files(source = source_F)
                        extract_F_KO = extract(files_F, Body_Part = Body_Part1)
                        F_KO_list.append(extract_F_KO)
                    elif user_in_F == 'n':
                        fill4 = [0]
                        intermediate_list4 = []
                        for i in range(0, 24):
                            intermediate_list4.append(fill4[0])
                        F_KO_list.append(intermediate_list4)
                    else:
                        pass
                    set_day += 1
                
                Data_Frame_Feeder_WT_M = [M_WT_list[i] for i in range(len(M_WT_list))]
                Data_Frame_Feeder_WT_F = [F_WT_list[i] for i in range(len(F_WT_list))] 
                Data_Frame_Feeder_KO_M = [M_KO_list[i] for i in range(len(M_KO_list))]
                Data_Frame_Feeder_KO_F = [F_KO_list[i] for i in range(len(F_KO_list))]
                
                reorder = [13, 14, 15, 16, 17, 18 ,19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12]
                
                Data_Frame_reorder_WT_M = []
                Data_Frame_reorder_WT_F = []
                Data_Frame_reorder_KO_M = []
                Data_Frame_reorder_KO_F = []
                
                day_F = 0
                while day_F < number_of_days_Female:
                    for i in reorder:
                        if len(Data_Frame_Feeder_WT_F[day_F]) != 0:
                            Data_Frame_reorder_WT_F.append(Data_Frame_Feeder_WT_F[day_F][i])
                        else:
                            pass
                        if len(Data_Frame_Feeder_KO_F[day_F]) != 0:
                            Data_Frame_reorder_KO_F.append(Data_Frame_Feeder_KO_F[day_F][i])
                        else:
                            pass
                    day_F += 1
                day_M = 0
                while day_M < number_of_days_Male:
                    for i in reorder:
                        if len(Data_Frame_Feeder_WT_M[day_M]) != 0:
                            Data_Frame_reorder_WT_M.append(Data_Frame_Feeder_WT_M[day_M][i])
                        else:
                            pass
                        if len(Data_Frame_Feeder_KO_M[day_M]) != 0: 
                            Data_Frame_reorder_KO_M.append(Data_Frame_Feeder_KO_M[day_M][i])
                        else:
                            pass
                    day_M += 1
                
               
                Data_Frame_reorder_WT_M = split_data(Data_Frame_reorder_WT_M, split_by_hours = number_of_days_Male)
                Data_Frame_reorder_KO_M = split_data(Data_Frame_reorder_KO_M, split_by_hours = number_of_days_Male)
               
                Data_Frame_reorder_WT_F = split_data(Data_Frame_reorder_WT_F, split_by_hours = number_of_days_Female)
                Data_Frame_reorder_KO_F = split_data(Data_Frame_reorder_KO_F, split_by_hours = number_of_days_Female)
                
                print(Data_Frame_reorder_WT_M)
                total_df_M = pd.DataFrame(data = {"Male, WT":Data_Frame_reorder_WT_M,
                                                  "Male, KO":Data_Frame_reorder_KO_M})
                total_df_F = pd.DataFrame(data = {"Female, WT":Data_Frame_reorder_WT_F,
                                                  "Female, KO":Data_Frame_reorder_KO_F})
        
                total_df_invert_M = total_df_M.transpose()
                total_df_invert_F = total_df_F.transpose()
                
                for i in range(0, number_of_days_Male):
                    total_df_invert_M = total_df_invert_M.rename(columns={i:("Day {}".format((i+1)))})
                for i in range(0, number_of_days_Female):
                    total_df_invert_F = total_df_invert_F.rename(columns={i:("Day {}".format((i+1)))})
                    
                    
                print(total_df_invert_M["Day 1"]["Male, WT"][0:6])
        
        fig, axes = mp.subplots(nrows=2, ncols=5, figsize = (26, 10))
        control_counter = 0
        timing = [i for i in range(0, 24)]
        #print(Data_Frame_reorder_KO_M)
        def multiplot_graph(WT_Data = total_df_invert_M["Day 1"]["Male, WT"], KO_Data = total_df_invert_M["Day 1"]["Male, KO"],
                            timing = timing, lower_ylim = 200000, upper_ylim1 = 215000, upper_ylim2 = 225000,
                            axis = axes[0,0], animal_label = "Male, Day 1", legend = 1):
            while_control = 0
            hourly_counter = 0
            while while_control < 1:
                for x, y in zip(WT_Data, KO_Data):
                    hourly_counter += 1
                    if (x > int(lower_ylim)) or (y > int(lower_ylim)):
                        if (all(x1 > 0 for x1 in WT_Data)) and (all(x2 > 0 for x2 in KO_Data)):
                            ax1 = axis
                            split = make_axes_locatable(ax1)
                            ax1_2 = split.new_vertical(size = '100%', pad = 0.3)
                            fig.add_axes(ax1_2)
                            
                            ax1.plot(timing, WT_Data, color = '#000000')
                            ax1.plot(timing, KO_Data, color = 'red')
                            if (axis == axes[0,0]) or (axis == axes[1,0]):
                                ax1.set_ylabel("Euclidean Distance", fontweight = 'bold')
                            ax1.set_ylim(0, int(lower_ylim))
                            ax1.spines['top'].set_visible(False)
                            ax1.set_xticklabels(["spacer", "1pm", "6pm", "11pm", "4am", "9am"])
                            labels = ["Wild Type", "Knock Out", "Dark phase: 7pm-7am"]
                            lines = [Line2D([0], [0], color = '#000000', linewidth = 3, linestyle = '-'), 
                                                Line2D([0], [0], color = 'red', linewidth = 3, linestyle = '-'),
                                                Line2D([0], [0], color = 'blue', linewidth = 3, linestyle = ':')]
                            
                            ax1_2.plot(timing, WT_Data, color = '#000000')
                            ax1_2.plot(timing, KO_Data, color = 'red')
                            ax1_2.set_ylim(int(upper_ylim1), int(upper_ylim2))
                            ax1_2.spines['bottom'].set_visible(False)
            
                            ax1.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                            ax1_2.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                            ax1.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                            ax1_2.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                            
                            if legend == 1:    
                                ax1_2.legend(lines, labels, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad =0.)
                            
                            ax1_2.tick_params(bottom = False, labelbottom = False)
                            ax1_2.get_xaxis().set_visible(False)
                            
                            ax1_2.set_title(animal_label, fontsize = 8)
                            
                            d = 0.015
                            kwargs = dict(transform = ax1_2.transAxes, color = 'k', clip_on = False)
                            ax1_2.plot((-d, +d), (-d, +d), **kwargs)
                            ax1_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                            
                            kwargs.update(transform = ax1.transAxes, color = 'k', clip_on = False)
                            ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                            ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                            
                            while_control += 1
                            if while_control == 1:
                                break
                        else:
                            if (all(x1 > 0 for x1 in WT_Data)):
                                ax1 = axis
                                split = make_axes_locatable(ax1)
                                ax1_2 = split.new_vertical(size = '100%', pad = 0.3)
                                fig.add_axes(ax1_2)
                                
                                ax1.plot(timing, WT_Data, color = '#000000')
                                if (axis == axes[0,0]) or (axis == axes[1,0]):
                                    ax1.set_ylabel("Euclidean Distance", fontweight = 'bold')
                                ax1.set_ylim(0, int(lower_ylim))
                                ax1.spines['top'].set_visible(False)
                                ax1.set_xticklabels(["spacer", "1pm", "6pm", "11pm", "4am", "9am"])
                                labels = ["Wild Type", "Knock Out", "Dark phase: 7pm-7am"]
                                lines = [Line2D([0], [0], color = '#000000', linewidth = 3, linestyle = '-'), 
                                                Line2D([0], [0], color = 'red', linewidth = 3, linestyle = '-'),
                                                Line2D([0], [0], color = 'blue', linewidth = 3, linestyle = ':')]
                                
                                ax1_2.plot(timing, WT_Data, color = '#000000')
                                ax1_2.set_ylim(int(upper_ylim1), int(upper_ylim2))
                                ax1_2.spines['bottom'].set_visible(False)
                
                                ax1.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1_2.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1_2.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                
                                if legend == 1:    
                                    ax1_2.legend(lines, labels, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad =0.)
                                
                                ax1_2.tick_params(bottom = False, labelbottom = False)
                                ax1_2.get_xaxis().set_visible(False)
                                
                                ax1_2.set_title(animal_label, fontsize = 8)
                                
                                d = 0.015
                                kwargs = dict(transform = ax1_2.transAxes, color = 'k', clip_on = False)
                                ax1_2.plot((-d, +d), (-d, +d), **kwargs)
                                ax1_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                                
                                kwargs.update(transform = ax1.transAxes, color = 'k', clip_on = False)
                                ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                                ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                                
                                while_control += 1
                                if while_control == 1:
                                    break
                                
                            elif (all(x2 > 0 for x2 in KO_Data)):
                                ax1 = axis
                                split = make_axes_locatable(ax1)
                                ax1_2 = split.new_vertical(size = '100%', pad = 0.3)
                                fig.add_axes(ax1_2)
                            
                                ax1.plot(timing, KO_Data, color = 'red')
                                if (axis == axes[0,0]) or (axis == axes[1,0]):
                                    ax1.set_ylabel("Euclidean Distance", fontweight = 'bold')
                                ax1.set_ylim(0, int(lower_ylim))
                                ax1.spines['top'].set_visible(False)
                                ax1.set_xticklabels(["spacer", "1pm", "6pm", "11pm", "4am", "9am"])
                                labels = ["Wild Type", "Knock Out", "Dark phase: 7pm-7am"]
                                lines = [Line2D([0], [0], color = '#000000', linewidth = 3, linestyle = '-'), 
                                                Line2D([0], [0], color = 'red', linewidth = 3, linestyle = '-'),
                                                Line2D([0], [0], color = 'blue', linewidth = 3, linestyle = ':')]

                                ax1_2.plot(timing, KO_Data, color = 'r', label = "Knock Out")
                                ax1_2.set_ylim(int(upper_ylim1), int(upper_ylim2))
                                ax1_2.spines['bottom'].set_visible(False)
                
                                ax1.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1_2.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1_2.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                
                                if legend == 1:    
                                    ax1_2.legend(lines, labels, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad =0.)
                                
                                ax1_2.tick_params(bottom = False, labelbottom = False)
                                ax1_2.get_xaxis().set_visible(False)
                                
                                ax1_2.set_title(animal_label, fontsize = 8)
                                
                                d = 0.015
                                kwargs = dict(transform = ax1_2.transAxes, color = 'k', clip_on = False)
                                ax1_2.plot((-d, +d), (-d, +d), **kwargs)
                                ax1_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                                
                                kwargs.update(transform = ax1.transAxes, color = 'k', clip_on = False)
                                ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                                ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                                
                                while_control += 1
                                if while_control == 1:
                                    break
                        
                    if (x <= int(lower_ylim) or y <= int(lower_ylim)) and ((hourly_counter == len(WT_Data)) or (hourly_counter == len(KO_Data))):
                        if (all(x1 > 0 for x1 in WT_Data)) and (all(x2 > 0 for x2 in KO_Data)): 
                            ax1 = axis
                            ax1.plot(timing, WT_Data, color = '#000000')
                            ax1.plot(timing, KO_Data, color = 'red')
                            labels = ["Wild Type", "Knock Out", "Dark phase: 7pm-7am"]
                            lines = [Line2D([0], [0], color = '#000000', linewidth = 3, linestyle = '-'), 
                                                Line2D([0], [0], color = 'red', linewidth = 3, linestyle = '-'),
                                                Line2D([0], [0], color = 'blue', linewidth = 3, linestyle = ':')]
                            ax1.set_title(animal_label, fontsize = 8)
                            
                            if (axis == axes[0,0]) or (axis == axes[1,0]):
                                ax1.set_ylabel("Euclidean Distance", fontweight = 'bold')
                                
                            ax1.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                            ax1.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                            ax1.set_ylim(0, lower_ylim)
                            ax1.set_xlim(-1.16, 24.3)
                            ax1.set_xticklabels(["spacer", "1pm", "6pm", "11pm", "4am", "9am"])
                        
                            if legend == 1:    
                                ax1.legend(lines, labels, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad =0.)
                            
                            print("done")
    
                        
                            while_control += 1
                            if while_control == 1:
                                break
                        else:
                            if all(x1 > 0 for x1 in WT_Data):
                                ax1 = axis
                                ax1.plot(timing, WT_Data, color = '#000000')
                                labels = ["Wild Type", "Knock Out", "Dark phase: 7pm-7am"]
                                lines = [Line2D([0], [0], color = '#000000', linewidth = 3, linestyle = '-'), 
                                                Line2D([0], [0], color = 'red', linewidth = 3, linestyle = '-'),
                                                Line2D([0], [0], color = 'blue', linewidth = 3, linestyle = ':')]
                                ax1.set_title(animal_label, fontsize = 8)
                                
                                if (axis == axes[0,0]) or (axis == axes[1,0]):
                                    ax1.set_ylabel("Euclidean Distance", fontweight = 'bold')
                                    
                                ax1.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1.set_ylim(0, lower_ylim)
                                ax1.set_xlim(-1.16, 24.3)
                                ax1.set_xticklabels(["spacer", "1pm", "6pm", "11pm", "4am", "9am"])
                            
                                if legend == 1:    
                                    ax1.legend(lines, labels, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad =0.)
                                
                                print("done")
        
                            
                                while_control += 1
                                if while_control == 1:
                                    break
                            
                            elif all(x2 > 0 for x2 in KO_Data):
                                ax1 = axis
                                ax1.plot(timing, KO_Data, color = 'red')
                                
                                labels = ["Wild Type", "Knock Out", "Dark phase: 7pm-7am"]
                                lines = [Line2D([0], [0], color = '#000000', linewidth = 3, linestyle = '-'), 
                                                Line2D([0], [0], color = 'red', linewidth = 3, linestyle = '-'),
                                                Line2D([0], [0], color = 'blue', linewidth = 3, linestyle = ':')]
                                ax1.set_title(animal_label, fontsize = 8)
                                
                                if (axis == axes[0,0]) or (axis == axes[1,0]):
                                    ax1.set_ylabel("Euclidean Distance", fontweight = 'bold')
                                    
                                ax1.axvline(6, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1.axvline(18, color = 'blue', zorder = 2.5, linestyle = 'dashed')
                                ax1.set_ylim(0, lower_ylim)
                                ax1.set_xlim(-1.16, 24.3)
                                ax1.set_xticklabels(["spacer", "1pm", "6pm", "11pm", "4am", "9am"])
        
                            
                                if legend == 1:    
                                    ax1.legend(lines, labels, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad =0.)
                                
                                #print("done")
        
                            
                                while_control += 1
                                if while_control == 1:
                                    break
        
        Master_I_list = []
        Master_I_KO_list = []
        
        light_phase_data_WT = []
        light_phase_data_KO = []
        try:
            plot1 = multiplot_graph(legend = 0)
            xlim = axes[0,0].get_xlim()
            
            ############
            #Darkphase
            ############
            #y in [x:y] is not included --> [x, y) 
            reorder_reset = reorder[6:19]
            #WT 
            function = line_eqn(reorder_reset, total_df_invert_M["Day 1"]["Male, WT"][6:19])
            integrate = integrator(function["slope"], function["Intercept"], reorder_reset)
            Master_I_list.append("Male WT, Day 1: {}".format(integrate[1]))
            #KO
            function = line_eqn(reorder_reset, total_df_invert_M["Day 1"]["Male, KO"][6:19])
            integrate = integrator(function["slope"], function["Intercept"], reorder_reset)
            Master_I_KO_list.append("Male KO, Day 1: {}".format(integrate[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_M["Day 1"]["Male, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_M["Day 1"]["Male, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Male WT, Day 1: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_M["Day 1"]["Male, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_M["Day 1"]["Male, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Male KO, Day 1: {}".format(Append_2))
     
        except KeyError:
            Master_I_list.append("Male WT, Day 1: {}".format(np.NaN))
            Master_I_KO_list.append("Male KO, Day 1: {}".format(np.NaN))
            
            light_phase_data_WT.append("Male WT, Day 1: {}".format(np.nan))
            light_phase_data_KO.append("Male KO, Day 1: {}".format(np.nan))
            
            pass
        except IndexError:
            pass
        
        try: 
            plot2 = multiplot_graph(WT_Data = total_df_invert_F["Day 1"]["Female, WT"], KO_Data = total_df_invert_F["Day 1"]["Female, KO"],
                                   timing = timing,
                                   axis = axes[1,0], animal_label = "Female, Day 1", legend = 0)
            ############
            #Darkphase
            ############
            reorder_reset = reorder[6:19]
            #WT
            function2 = line_eqn(reorder_reset, total_df_invert_F["Day 1"]["Female, WT"][6:19])
            integrate2 = integrator(function2["slope"], function2["Intercept"], reorder_reset)
            Master_I_list.append("Female WT, Day 1: {}".format(integrate2[1]))
            #KO
            function2 = line_eqn(reorder_reset, total_df_invert_F["Day 1"]["Female, KO"][6:19])
            integrate2 = integrator(function2["slope"], function2["Intercept"], reorder_reset)
            Master_I_KO_list.append("Female KO, Day 1: {}".format(integrate2[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_F["Day 1"]["Female, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_F["Day 1"]["Female, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Female WT, Day 1: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_F["Day 1"]["Female, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_F["Day 1"]["Female, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Female KO, Day 1: {}".format(Append_2))
            
        except KeyError:
            if axes[1,0].get_xlim() != xlim:
                axes[1,0].remove()
            Master_I_list.append("Female WT, Day 1: {}".format(np.NaN))
            Master_I_KO_list.append("Female KO, Day 1: {}".format(np.NaN))
            
            light_phase_data_WT.append("Female WT, Day 1: {}".format(np.nan))
            light_phase_data_KO.append("Female KO, Day 1: {}".format(np.nan))
            pass
        except IndexError:
            print("Error2")
            pass
        
        try:
            plot3 = multiplot_graph(WT_Data = total_df_invert_M["Day 2"]["Male, WT"], KO_Data = total_df_invert_M["Day 2"]["Male, KO"],
                                   timing = timing, 
                                   axis = axes[0,1], animal_label = "Male, Day 2", legend = 0)
            reorder_reset = reorder[6:19]
            #WT
            function3 = line_eqn(reorder_reset, total_df_invert_M["Day 2"]["Male, WT"][6:19])
            integrate3 = integrator(function3["slope"], function3["Intercept"], reorder_reset)
            Master_I_list.append("Male WT, Day 2: {}".format(integrate3[1]))
            #KO
            function3 = line_eqn(reorder_reset, total_df_invert_M["Day 2"]["Male, KO"][6:19])
            integrate3 = integrator(function3["slope"], function3["Intercept"], reorder_reset)
            Master_I_KO_list.append("Male KO, Day 2: {}".format(integrate3[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_M["Day 2"]["Male, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_M["Day 2"]["Male, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Male WT, Day 2: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_M["Day 2"]["Male, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_M["Day 2"]["Male, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Male KO, Day 2: {}".format(Append_2))


        except KeyError:
            if axes[0,1].get_xlim() != xlim:
                axes[0,1].remove()
            
            Master_I_list.append("Male WT, Day 2: {}".format(np.NaN))
            Master_I_KO_list.append("Male KO, Day 2: {}".format(np.NaN))
            
            light_phase_data_WT.append("Male WT, Day 2: {}".format(np.nan))
            light_phase_data_KO.append("Male KO, Day 2: {}".format(np.nan))
            
            
            pass
        except IndexError:
            pass
        
        try: 
            plot4 = multiplot_graph(WT_Data = total_df_invert_F["Day 2"]["Female, WT"], KO_Data = total_df_invert_F["Day 2"]["Female, KO"],
                                   timing = timing,
                                   axis = axes[1,1], animal_label = "Female, Day 2", legend = 0)
            reorder_reset = reorder[6:19]
            #WT
            function4 = line_eqn(reorder_reset, total_df_invert_F["Day 2"]["Female, WT"][6:19])
            integrate4 = integrator(function4["slope"], function4["Intercept"], reorder_reset)
            Master_I_list.append("Female WT, Day 2: {}".format(integrate4[1]))
            #KO
            function4 = line_eqn(reorder_reset, total_df_invert_F["Day 2"]["Female, KO"][6:19])
            integrate4 = integrator(function4["slope"], function4["Intercept"], reorder_reset)
            Master_I_KO_list.append("Female KO, Day 2: {}".format(integrate4[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_F["Day 2"]["Female, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_F["Day 2"]["Female, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Female WT, Day 2: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_F["Day 2"]["Female, KO"][0:6])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_F["Day 2"]["Female, KO"][19:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Female KO, Day 2: {}".format(Append_2))
        
        except KeyError:
            if axes[1,1].get_xlim() != xlim:
                axes[1,1].remove()
            Master_I_list.append("Female WT, Day 2: {}".format(np.NaN))
            Master_I_KO_list.append("Female KO, Day 2: {}".format(np.NaN))
            
            light_phase_data_WT.append("Female WT, Day 2: {}".format(np.nan))
            light_phase_data_KO.append("Female KO, Day 2: {}".format(np.nan))
            pass

        except IndexError:
            print("Error2")
            pass
        
        try:
            plot5 = multiplot_graph(WT_Data = total_df_invert_M["Day 3"]["Male, WT"], KO_Data = total_df_invert_M["Day 3"]["Male, KO"],
                                   timing = timing, 
                                   axis = axes[0,2], animal_label = "Male, Day 3", legend = 0)
            reorder_reset = reorder[6:19]
            #WT
            function5 = line_eqn(reorder_reset, total_df_invert_M["Day 3"]["Male, WT"][6:19])
            integrate5 = integrator(function5["slope"], function5["Intercept"], reorder_reset)
            Master_I_list.append("Male WT, Day 3: {}".format(integrate5[1]))
            #KO
            function5 = line_eqn(reorder_reset, total_df_invert_M["Day 3"]["Male, KO"][6:19])
            integrate5 = integrator(function5["slope"], function5["Intercept"], reorder_reset)
            Master_I_KO_list.append("Male KO, Day 3: {}".format(integrate5[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_M["Day 3"]["Male, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_M["Day 3"]["Male, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Male WT, Day 3: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_M["Day 3"]["Male, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_M["Day 3"]["Male, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Male KO, Day 3: {}".format(Append_2))
            
        except KeyError:
            if axes[0,2].get_xlim() != xlim:
                axes[0,2].remove()
            Master_I_list.append("Male WT, Day 3: {}".format(np.NaN))
            Master_I_KO_list.append("Male KO, Day 3: {}".format(np.NaN))
            
            light_phase_data_WT.append("Male WT, Day 3: {}".format(np.nan))
            light_phase_data_KO.append("Male KO, Day 3: {}".format(np.nan))
            
            pass
        try: 
            plot6 = multiplot_graph(WT_Data = total_df_invert_F["Day 3"]["Female, WT"], KO_Data = total_df_invert_F["Day 3"]["Female, KO"],
                                   timing = timing,
                                   axis = axes[1,2], animal_label = "Female, Day 3", legend = 0)
            reorder_reset = reorder[6:19]
            #WT
            function6 = line_eqn(reorder_reset, total_df_invert_F["Day 3"]["Female, WT"][6:19])
            integrate6 = integrator(function6["slope"], function6["Intercept"], reorder_reset)
            Master_I_list.append("Female WT, Day 3: {}".format(integrate6[1]))
            #KO
            function6 = line_eqn(reorder_reset, total_df_invert_F["Day 3"]["Female, KO"][6:19])
            integrate6 = integrator(function6["slope"], function6["Intercept"], reorder_reset)
            Master_I_KO_list.append("Female KO, Day 3: {}".format(integrate6[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_F["Day 3"]["Female, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_F["Day 1"]["Female, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Female WT, Day 3: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_F["Day 3"]["Female, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_F["Day 3"]["Female, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Female KO, Day 3: {}".format(Append_2))
            
        except KeyError:
            if axes[1,2].get_xlim() != xlim:
                axes[1,2].remove()    
            Master_I_list.append("Female WT, Day 3: {}".format(np.NaN))
            Master_I_KO_list.append("Female KO, Day 3: {}".format(np.NaN))
            
            light_phase_data_WT.append("Female WT, Day 3: {}".format(np.nan))
            light_phase_data_KO.append("Female KO, Day 3: {}".format(np.nan))
            pass    
          
        except IndexError:
            print("Error2")
            pass
        
        try:
            plot7 = multiplot_graph(WT_Data = total_df_invert_M["Day 4"]["Male, WT"], KO_Data = total_df_invert_M["Day 4"]["Male, KO"],
                                   timing = timing, 
                                   axis = axes[0,3], animal_label = "Male, Day 4", legend = 1)
            reorder_reset = reorder[6:19]
            #WT
            function7 = line_eqn(reorder_reset, total_df_invert_M["Day 4"]["Male, WT"][6:19])
            integrate7 = integrator(function7["slope"], function7["Intercept"], reorder_reset)
            Master_I_list.append("Male WT, Day 4: {}".format(integrate7[1]))
            #KO
            function7 = line_eqn(reorder_reset, total_df_invert_M["Day 4"]["Male, KO"][6:19])
            integrate7 = integrator(function7["slope"], function7["Intercept"], reorder_reset)
            Master_I_KO_list.append("Male KO, Day 4: {}".format(integrate7[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_M["Day 4"]["Male, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_M["Day 4"]["Male, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Male WT, Day 4: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_M["Day 4"]["Male, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_M["Day 4"]["Male, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Male KO, Day 4: {}".format(Append_2))

        except KeyError:
            if axes[0,3].get_xlim() != xlim:
                axes[0,3].remove()
            
            Master_I_list.append("Male WT, Day 4: {}".format(np.NaN))
            Master_I_KO_list.append("Male KO, Day 4: {}".format(np.NaN))
            
            light_phase_data_WT.append("Male WT, Day 4: {}".format(np.nan))
            light_phase_data_KO.append("Male KO, Day 4: {}".format(np.nan))
            
            pass
        
        try: 
            plot8 = multiplot_graph(WT_Data = total_df_invert_F["Day 4"]["Female, WT"], KO_Data = total_df_invert_F["Day 4"]["Female, KO"],
                                   timing = timing,
                                   axis = axes[1,3], animal_label = "Female, Day 4", legend = 0)
            reorder_reset = reorder[6:19]
            #WT
            function8 = line_eqn(reorder_reset, total_df_invert_F["Day 4"]["Female, WT"][6:19])
            integrate8 = integrator(function8["slope"], function8["Intercept"], reorder_reset)
            Master_I_list.append("Female WT, Day 4: {}".format(integrate8[1]))
            #KO
            function8 = line_eqn(reorder_reset, total_df_invert_F["Day 4"]["Female, KO"][6:19])
            integrate8 = integrator(function8["slope"], function8["Intercept"], reorder_reset)
            Master_I_KO_list.append("Female KO, Day 4: {}".format(integrate8[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_F["Day 4"]["Female, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_F["Day 4"]["Female, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Female WT, Day 4: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_F["Day 4"]["Female, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_F["Day 4"]["Female, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Female KO, Day 4: {}".format(Append_2))
            
        except KeyError:
            if axes[1,3].get_xlim() != xlim:
                axes[1,3].remove()
            Master_I_list.append("Female WT, Day 4: {}".format(np.NaN))
            Master_I_KO_list.append("Female KO, Day 4: {}".format(np.NaN))
            
            light_phase_data_WT.append("Female WT, Day 4: {}".format(np.nan))
            light_phase_data_KO.append("Female KO, Day 4: {}".format(np.nan))
            
            pass
        except IndexError:
            print("Error2")
            pass
        
        try:
            plot9 = multiplot_graph(WT_Data = total_df_invert_M["Day 5"]["Male, WT"], KO_Data = total_df_invert_M["Day 5"]["Male, KO"],
                                   timing = timing, 
                                   axis = axes[0,4], animal_label = "Male, Day 5", legend = 0)
            reorder_reset = reorder[6:19]
            #WT
            function9 = line_eqn(reorder_reset, total_df_invert_M["Day 5"]["Male, WT"][6:19])
            integrate9 = integrator(function9["slope"], function9["Intercept"], reorder_reset)
            Master_I_list.append("Male WT, Day 5: {}".format(integrate9[1]))
            #KO
            function9 = line_eqn(reorder_reset, total_df_invert_M["Day 5"]["Male, KO"][6:19])
            integrate9 = integrator(function9["slope"], function9["Intercept"], reorder_reset)
            Master_I_KO_list.append("Male KO, Day 5: {}".format(integrate9[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_M["Day 5"]["Male, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_M["Day 5"]["Male, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Male WT, Day 5: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_M["Day 5"]["Male, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_M["Day 5"]["Male, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Male KO, Day 5: {}".format(Append_2))
            
        except KeyError:
            if axes[0,4].get_xlim() != xlim:
                axes[0,4].remove()
            Master_I_list.append("Male WT, Day 5: {}".format(np.NaN))
            Master_I_KO_list.append("Male KO, Day 5: {}".format(np.NaN))
            
            light_phase_data_WT.append("Male WT, Day 5: {}".format(np.nan))
            light_phase_data_KO.append("Male KO, Day 5: {}".format(np.nan))
            
            pass

        try:
            plot10 = multiplot_graph(WT_Data = total_df_invert_F["Day 5"]["Female, WT"], KO_Data = total_df_invert_F["Day 5"]["Female, KO"],
                                   timing = timing, 
                                   axis = axes[1,4], animal_label = "Female, Day 5", legend = 0)
            reorder_reset = reorder[6:19]
            #WT
            function10 = line_eqn(reorder_reset, total_df_invert_F["Day 5"]["Female, WT"][6:19])
            integrate10 = integrator(function10["slope"], function10["Intercept"], reorder_reset)
            Master_I_list.append("Female WT, Day 5: {}".format(integrate10[1]))
            #KO
            function10 = line_eqn(reorder_reset, total_df_invert_F["Day 5"]["Female, KO"][6:19])
            integrate10 = integrator(function10["slope"], function10["Intercept"], reorder_reset)
            Master_I_KO_list.append("Female KO, Day 5: {}".format(integrate10[1]))
            
            ############
            #Lightphase
            ############
            reorder_reset_1 = reorder[0:7]
            reorder_reset_2 = reorder[18:24]
            # #WT
            function_light_phase_1 = line_eqn(reorder_reset_1, total_df_invert_F["Day 5"]["Female, WT"][0:7])
            integrate_light_phase_1 = integrator(function_light_phase_1["slope"], function_light_phase_1["Intercept"], reorder_reset_1)            
            function_light_phase_2 = line_eqn(reorder_reset_2, total_df_invert_F["Day 5"]["Female, WT"][18:24])
            integrate_light_phase_2 = integrator(function_light_phase_2["slope"], function_light_phase_2["Intercept"], reorder_reset_2)
            Append = (integrate_light_phase_1[1] + integrate_light_phase_2[1])
            light_phase_data_WT.append("Female WT, Day 5: {}".format(Append))
            #KO
            function_light_phase_3 = line_eqn(reorder_reset_1, total_df_invert_F["Day 5"]["Female, KO"][0:7])
            integrate_light_phase_3 = integrator(function_light_phase_3["slope"], function_light_phase_3["Intercept"], reorder_reset_1)            
            function_light_phase_4 = line_eqn(reorder_reset_2, total_df_invert_F["Day 5"]["Female, KO"][18:24])
            integrate_light_phase_4 = integrator(function_light_phase_4["slope"], function_light_phase_4["Intercept"], reorder_reset_2)
            Append_2 = (integrate_light_phase_3[1] + integrate_light_phase_4[1])
            light_phase_data_KO.append("Female KO, Day 5: {}".format(Append_2))
        
        except KeyError:
            if axes[1,4].get_xlim() != xlim:
                axes[1,4].remove()
            Master_I_list.append("Female WT, Day 5: {}".format(np.NaN))
            Master_I_KO_list.append("Female KO, Day 5: {}".format(np.NaN))
            
            light_phase_data_WT.append("Female WT, Day 5: {}".format(np.nan))
            light_phase_data_KO.append("Female KO, Day 5: {}".format(np.nan))
            
            pass
   
    try:
        L_D_dataframe = pd.DataFrame(data={"WT_Female_darkphase":[Master_I_list[x] for x in range(len(Master_I_list)) if x % 2 != 0],
                                          "WT_Male_darkphase":[Master_I_list[y] for y in range(len(Master_I_list)) if y % 2 == 0],
                                          "KO_Female_darkphase":[Master_I_KO_list[n] for n in range(len(Master_I_KO_list)) if n % 2 != 0],
                                          "KO_Male_darkphase":[Master_I_KO_list[m] for m in range(len(Master_I_KO_list)) if m % 2 == 0],
                                          "WT_Female_lightphase":[light_phase_data_WT[i] for i in range(len(light_phase_data_WT)) if i % 2 != 0],
                                          "WT_Male_lightphase":[light_phase_data_WT[j] for j in range(len(light_phase_data_WT)) if j % 2 == 0],
                                          "KO_Female_lightphase":[light_phase_data_KO[z] for z in range(len(light_phase_data_KO)) if z % 2 != 0],
                                          "KO_Male_lightphase":[light_phase_data_KO[c] for c in range(len(light_phase_data_KO)) if c % 2 == 0]})
        
        # D_DataFrame = pd.DataFrame(data={"WT_Female_darkphase":[Master_I_list[x] for x in range(len(Master_I_list)) if x % 2 != 0],
        #                                   "WT_Male_darkphase":[Master_I_list[y] for y in range(len(Master_I_list)) if y % 2 == 0],
        #                                   "KO_Female_darkphase":[Master_I_KO_list[n] for n in range(len(Master_I_KO_list)) if n % 2 != 0],
        #                                   "KO_Male_darkphase":[Master_I_KO_list[m] for m in range(len(Master_I_KO_list)) if m % 2 == 0]})
        
        print(L_D_dataframe)
        L_D_dataframe.to_csv('L_D_DataFrame.csv')
        for ax in fig.get_axes():
            print(ax.get_xlim())
        
        slicenum = int(number_of_days_Female + number_of_days_Male)
        for ax in fig.get_axes()[0:slicenum]:
            ax.set_xlabel("Time", fontweight = 'bold')
    
        mp.show()
    except NameError:
        pass

    
        