import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

'''
This function removes rows containing NaNs from a 
DataFrame. Works for both 1D and 2D DataFrames.
'''
def remove_nan(data):
    if len(data.shape)>1:
        return data[(~np.isnan(data)).all(1)]
    else:
        return data[~np.isnan(data)]

'''
This function removes rows with values below or equal to 0 from a 
DataFrame. Works for both 1D and 2D DataFrames.
'''
def select_above_zero(data):
    if len(data.shape)>1:
        return data[(data>0).all(1)]
    else:
        return data[data>0]

'''
This function removes rows with values below 0 from a 
DataFrame. Works for both 1D and 2D DataFrames.
'''
def select_nonnegative(data):
    if len(data.shape)>1:
        return data[(data>=0).all(1)]
    else:
        return data[data>=0]

'''
Plot a histogram based on a 1D DataFrame
'''
def plot_one_histogram(data, label="", xlabel="", ylabel="", title=""):
    max_val = int(max(data))
    min_val = int(min(data))
    bins = np.linspace(min_val, max_val, min(max_val-min_val,100))
    fig = plt.figure(figsize=(7,7))
    plt.hist(data, bins, alpha=0.5, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right")

'''
Plot two overlapping histograms based on two 1D DataFrames
'''
def plot_two_histograms(data1, data2, label1="", label2="", xlabel="", ylabel="", title=""):
    max_val = int(max(max(data1),max(data2)))
    min_val = int(min(min(data1),min(data2)))
    # the numpy linspace function to create an array of evenly spaced numbers
    bins = np.linspace(min_val, max_val, min(max_val-min_val,100))
    # Let's create the matplotlib figure where we will plots the histograms.
    fig = plt.figure(figsize=(7,7))
    # Plot the histograms. We use pyplot's hist function and 
    plt.hist(data1, bins, alpha=0.5, label=label1)
    plt.hist(data2, bins, alpha=0.5, label=label2)
    # Label the plot.
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Plot a legend so that we can match the color of the histogram to the data.
    plt.legend(loc="upper right")
'''
This function takes in two arrays of equal length (xdata, ydata)
and plots them against each other in a scatterplot
'''
def scatterplot(xdata, ydata, xlabel="", ylabel="", title=""):
    fig = plt.figure(figsize=(7,7))
    plt.scatter(xdata, ydata)
    #plt.plot([0,data_max],[0,data_max])
    plt.xlim(min(xdata)-1,max(xdata)+1)
    plt.ylim(min(ydata)-1,max(ydata)+1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

'''
This function takes in two arrays of equal length (xdata, ydata)
and plots them against each other in a scatterplot
With a line with a given intercept and slope overlaid on top
'''
def scatterplot_with_line(xdata, ydata, slope, intercept, xlabel="", ylabel="", title=""):
    fig = plt.figure(figsize=(7,5))
    plt.scatter(xdata, ydata)
    x = np.linspace(min(xdata)-1,max(xdata)+1, 100)
    plt.plot(x, slope*x+intercept)
    plt.xlim(min(xdata)-1,max(xdata)+1)
    plt.ylim(min(ydata)-1,max(ydata)+1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def two_bar_charts(data1, data2, label1="", label2="", bar_labels=[]):
    labels = np.arange(4)
    fig = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(121) # 121 means we want 1 row, 2 columns, and that we want to plot on the first of these 2 subplots
    plt.title(label1)
    ax1.bar(labels, data1)
    ax2 = plt.subplot(122) # 122 means we want the 2nd plot of the 1 row, 2 columns figure.
    plt.title(label2)
    ax2.bar(labels, data2)
    ax1.set_xticks(labels+0.4)
    ax1.set_xticklabels(bar_labels)
    ax2.set_xticks(labels+0.4)
    ax2.set_xticklabels(bar_labels)

def one_bar_chart(data, label="", bar_labels=[]):
    labels = np.arange(4)
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(121) # 121 means we want 1 row, 2 columns, and that we want to plot on the first of these 2 subplots
    plt.title(label)
    ax.bar(labels, data)
    ax.set_xticks(labels+0.4)
    ax.set_xticklabels(bar_labels)

def one_pie_chart(data, label, pie_labels=[]):
    explode = (0, 0, 0, 0.1)
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(121)
    plt.title(label)
    ax1.pie(data, explode=explode, labels=pie_labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

def two_pie_charts(data1, data2, label1, label2, pie_labels=[]):
    explode = (0, 0, 0, 0.1)
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(30,10))
    ax1 = plt.subplot(121)
    plt.title("Mother's Education")
    ax1.pie(data1, explode=explode, labels=pie_labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    ax2 = plt.subplot(122)
    plt.title("Father's Education")
    ax2.pie(data2, explode=explode, labels=pie_labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.axis('equal')

'''
This function takes in the outcome and backtround DataFrame,
a list of desired background variables and a list desired outcome_vars,
and subselects them from the background and output frames.
It returns a single DataFrame containing the desired columns, where 
corresponding rows between the two DataFrames have been subselected.
'''
def pick_out_variables(background, output, background_vars, outcome_vars, remove_nans=False, remove_negatives=False):
    train_X = background.loc[outcome.index]
    new_frame = train_X[background_vars]
    new_frame[outcome_vars] = output[outcome_vars]
    if remove_nans:
        if len(new_frame.shape)>1:
            new_frame = new_frame[(~np.isnan(new_frame)).all(1)]
        else:
            new_frame = new_frame[~np.isnan(new_frame)]
    if remove_negatives:
        if len(new_frame.shape)>1:
            new_frame = new_frame[(new_frame>=0).all(1)]
        else:
            new_frame = new_frame[new_frame>=0]
    return new_frame