#import submodules from plotly
#Import plotly package
import plotly 

#We have generated a set 
#Sets location of plots in the notebook instead of in new window
plotly.offline.init_notebook_mode()


import plotly.plotly as py
from plotly.graph_objs import * 

def plot_RNAseq_barplot(batches):
    #calculate the number of samples for each system in the batches file
    df=batches['System'].value_counts()

    #Define the x and y values for the bar plot  
    data = [Bar(x=df.index,y=df.values)]

    #Label the axes
    layout=Layout(xaxis=dict(title='System'),yaxis=dict(title='Number of Samples'))

    #Draw the figure  
    fig=Figure(data=data,layout=layout)
    plotly.offline.iplot(fig)


def scree_plot(data):
    screeplotdata = [Bar(x=list(range(1,11)),y=data)]

    #Label the axes and add a plot title 
    screeplotlayout=Layout(title="Scree Plot",barmode="group",xaxis=dict(title='PC'),yaxis=dict(title='Fraction of variance explained'))

    #Draw the figure 
    fig=Figure(data=screeplotdata,layout=screeplotlayout)
    plotly.offline.iplot(fig)
