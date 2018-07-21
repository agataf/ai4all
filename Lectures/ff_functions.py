import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


'''
This function removes rows containing NaNs from a 
DataFrame. 
'''
def remove_nan(data):
    if len(data.shape)>1:
        return data[(~np.isnan(data)).all(1)]
    else:
        return data[~np.isnan(data)]

'''
This function removes rows with values below or equal to 0 from a 
DataFrame. 
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
This function takes in the backtround DataFrame,
a list of desired background variables,
and subselects them from the background frame.
It returns a single DataFrame containing the desired columns, where 
corresponding rows between the two DataFrames have been subselected.

The function also provides the options to remove nan variables and remove negative values.

Input arguments:

Required:
dataframe: the orginal DataFrame (to be reduced)
features: a list of column names to subselect from the dataframe

Additional (with default values filled in):
remove_nans: if True, remove rows containing NaN values
remove_negatives: if True, remove rows containing negative values


Output: 
a pandas dataframe containing only selected columns (features).
'''
def pick_ff_variables(dataframe, features, remove_nans=False, remove_negatives=False):
    # For exery feature inside the list of features, make sure it's contained in the columns
    for ft in features:
        if ft not in dataframe.columns:
            print("Feature " + ft + " is in NOT the columns - provide other features.")
    
    # select only the columns corresponding to desired features
    new_frame = dataframe[features]
    
    # option to remove NaNs
    if remove_nans:
        if len(new_frame.shape)>1:
            new_frame = new_frame[(~np.isnan(new_frame)).all(1)]
        else:
            new_frame = new_frame[~np.isnan(new_frame)]
    
    # option to remove negative values
    if remove_negatives:
        if len(new_frame.shape)>1:
            new_frame = new_frame[(new_frame>=0).all(1)]
        else:
            new_frame = new_frame[new_frame>=0]
            
    print("Data frame with ", new_frame.shape[0], " rows and ", new_frame.shape[1], "columns.")
    return new_frame


'''
Plot a histogram based on a DataFrame "data"

Input arguments:

Required:
data: the pandas DataFrame to be plotted

Additional (with default values filled in):
labels: a list of labels for each column of the DataFrame
xlabel: label for the x axis
ylabel: label for the y axis
title: plot title

Output: 
Histogram plotted directly in notebook
'''
def plot_histogram(data, labels=[], xlabel="", ylabel="", title=""):
    
    # Set the ranges for X and Y coorinates to be the lowest and highest values observed
    max_val = int(np.max(data.values))
    min_val = int(np.min(data.values))
    
    # choose ranges into which we'll "bin" (count) occurences of variables
    bins = np.linspace(min_val, max_val, min(max_val-min_val,50))
    
    # set figure size
    fig = plt.figure(figsize=(7,7))
    
    # two options for dataframes with one vs more than one column
    if len(data.shape) > 1:
        for i,column in enumerate(data.columns):
            # build the histogram from a given column, bin it
            # set how "see-though" it is (alpha parameters)
            # set the label which will be displayed witht his histogram
            plt.hist(data[column], bins, alpha=1/len(data.shape), label=labels[i])
    else:
        plt.hist(data.values, bins, alpha=0.5, label=labels[0])
        
    # Set x and y corrdinates labels, title    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # put the legend in the upper right corner
    plt.legend(loc="upper right")

'''
This function takes in two arrays of equal length (xdata, ydata)
and plots them against each other in a scatterplot

Input arguments:

Required:
xdata: a column of a Pandas DataFrame, data to be plotted against the x axis
ydata: a column of a Pandas DataFrame, data to be plotted against the y axis
(must be same length as xdata)

Additional (with default values filled in):
xlabel: label for the x axis (string)
ylabel: label for the y axis (string)
title: plot title (string)
plot_diagonal: if true (boolean)


Output: 
Scatterplot plotted directly in notebook
'''
def scatterplot(xdata, ydata, xlabel="", ylabel="", title="", plot_diagonal=False):
    
    # start new figure with sizes 7, 7 (try changing the numbers
    # to see the impact on the size of printed plot)
    fig = plt.figure(figsize=(7,7))
    
    # set the ranges for x and y axes to be between
    # the minimum value and the maximum value
    # (with an additional margin of 1 on each side for clarity)
    plt.xlim(min(xdata)-5,max(xdata)+5)
    plt.ylim(min(ydata)-5,max(ydata)+5)
    
    if plot_diagonal:
        x = np.linspace(min(xdata)-5,max(xdata)+5, 100)
        plt.plot(x, x)
    
    # plot the data
    plt.scatter(xdata, ydata)
    
    # set the selected labels for x and y axes, and title (empty by default)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

'''
This function converts a pandas 1D series to a numpy 2D array (for reference, see lecture 2.2),
which can be used with sklearn functions
'''
def pandas_to_2d_numpy(pandas_series):
    num_rows = pandas_series.shape[0]
    # cast the Pandas Series to a numpy array (because sklearn works with numpy)
    numpy_series = np.array(pandas_series)
    # expand the number of dimensions (from (d,0) to (d,1)) - 1d list to 2d list
    numpy_2d_series = numpy_series.reshape(num_rows,1)
    return numpy_2d_series

from mpl_toolkits.mplot3d import Axes3D

'''
This function takes in three arrays of equal length (xdata, ydata, zdata)
and plots them against each other in a 3D scatterplot

Required:
xdata: a column of a Pandas DataFrame, data to be plotted against the x axis
ydata: a column of a Pandas DataFrame, data to be plotted against the y axis
xdata: a column of a Pandas DataFrame, data to be plotted against the x axis
(must all be same length)

Additional (with default values filled in):
xlabel: label for the x axis (string)
ylabel: label for the y axis (string)
zlabel: label for the z axis (string)
title: plot title (string)
slope: 2d list with x and y slopes, to plot a line
intercept: a float or int with z-intercept, to plot a line
'''
def scatter_3d(xdata, ydata, zdata, xlabel="", ylabel="", zlabel="", title="", slope=None, intercept=None):
    
    # choose figure size
    fig = plt.figure(figsize=(7,7))
    
    # create a "subplot" and make it 3d
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('equal')
    
    # plot the data
    ax.scatter(xdata, ydata, zdata)
    
    # set the selected labels for x and y axes, and title (empty by default)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    # optionally: draw a line, when slope and intercept provided
    if (slope is not None) and (intercept is not None):
        x = np.linspace(min(xdata)-1,max(xdata)+1, 100)
        y = np.linspace(min(ydata)-1,max(ydata)+1, 100)
        ax.plot(x, y, slope[0]*x+slope[1]*y+intercept)


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
This function takes in a DataFrame and returns a correlation matrix heatmap for all variables
present in the frame.

Input arguments:

Required:
frame: a Pandas DataFrame, containing features whose correlations we're interested in

Output: 
Heatmap-colored correlation plot
'''
def get_corr_plot(frame):
    
    # calculate the correlations between all variables in the frame
    correlations = frame.astype(float).corr()
    
    # plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # set the color scale
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)

'''
This function takes in two arrays of equal length (xdata, ydata)
and plots them against each other in a scatterplot
With a line with a given intercept and slope overlaid on top
'''
def scatterplot_with_line(xdata, ydata, slope, intercept, xlabel="", ylabel="", title=""):
    fig = plt.figure(figsize=(7,5))
    plt.scatter(xdata, ydata)
    x = np.linspace(min(xdata)-5,max(xdata)+5, 100)
    plt.plot(x, slope*x+intercept,lw=2,c='r')
    plt.xlim(min(xdata)-5,max(xdata)+5)
    plt.ylim(min(ydata)-5,max(ydata)+5)
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
def pick_challenge_variables(background, output, background_vars, outcome_vars, remove_nans=False, remove_negatives=False):
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

