import random
import numpy as np
import pandas as pd 
import math 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
from plotnine import * 

red='#ff0000'
black='#000000'
blue='#0000ff'
green='#00ff00'
purple='#800080'
orange='#FFA500'
gray='#808080'
color_options=[red,blue,green,purple,orange,gray]

def generate_random_data(N):
    c1=np.random.normal([-2,2],0.5,([int(N/3),2]))
    c2=np.random.normal([0,1],0.5,([int(N/3),2]))
    c3=np.random.normal([2,2],0.5,([int(N/3),2]))
    A=list(c1[:,0])+list(c2[:,0])+list(c3[:,0])
    B=list(c1[:,1])+list(c2[:,1])+list(c3[:,1])
    return A,B


def get_color_assignments(cluster_assignments,num_points):
        #define the colors we will use for the plot
        if cluster_assignments==None:
                    return [black]*num_points
        else:
            colors=[]
            for c in cluster_assignments:
                colors.append(color_options[c])
            return colors

def plot(x_coords,
         y_coords,
         cluster_assignments=None,
         x_centroids=None,
         y_centroids=None):
    
    #Figure out the colors to use for plotting. 
    num_points=len(x_coords)
    colors=get_color_assignments(cluster_assignments,num_points)
    #Define the x and y values for the scatter plot
    data=pd.DataFrame({'x':x_coords,'y':y_coords,'cluster':colors})

    if x_centroids is None:
        return (ggplot(data,aes('x','y'))+
                geom_point(color=colors))
    else:
        num_clusters=len(x_centroids)
        centroids_data=pd.DataFrame({'x_centroids':x_centroids,'y_centroids':y_centroids})
        return (ggplot(data)+
                geom_point(aes('x_coords','y_coords'),color=colors)+
                geom_point(centroids_data,aes('x_centroids','y_centroids'),shape='X',size=10,color=color_options[0:num_clusters])+
                xlab('x')+
                ylab('y'))

def initialize_centroids(k,min_val,max_val):
    '''
    k -- the number of centroids to initialize. This is equal to the number of clusters to generate. 
    min_val -- the minimum allowed centroid coordinate (same for both x and y coordinates). 
    max_val -- the maximum allowed centroid coordinate (same for both x and y coordinates). 
    
    returns two lists: the x and y coordinates of the generated centroids. 
    '''
    x_centroids=[] 
    y_centroids=[] 
    for cluster in range(k): 
        x_centroids.append(random.uniform(min_val,max_val))
        y_centroids.append(random.uniform(min_val,max_val))
    print("x-coordinates of centroids:"+ str(x_centroids))
    print("y-coordinates of centroids:"+ str(y_centroids))
    return x_centroids,y_centroids 
    
    

def distance(x,y,x_centroid,y_centroid):
    ''' 
    x -- list containing the x-coordinates of the points in the dataset 
    y -- list containing the y-coordinates of the points in the dataset 
    x_centroid -- x-coordinate of the centroid 
    y_centroid -- y-coordinate of the centroid 
    returns a list of Euclidean distances from each point to the centroid. 
    '''
    distances=[] 
    for i in range(len(x)): 
        cur_x=x[i] 
        cur_y=y[i] 
        cur_distance=math.sqrt((cur_x-x_centroid)**2+(cur_y-y_centroid)**2)
        distances.append(cur_distance)
    return distances 

#Now, define a formula to assign a point to the nearest centroid, based on the distance calculated in the formula
#above. 
def assign_cluster(distances,num_points):
    cluster_assignments=[] 
    for point_index in range(num_points): 
        point_distances=[d[point_index] for d in distances]
        cluster_assignment=point_distances.index(min(point_distances))
        cluster_assignments.append(cluster_assignment)
    return cluster_assignments
    
    
def update_centroids(x,y,cluster_assignments,k):
    new_x_centroids=[] 
    new_y_centroids=[] 
    num_points=len(x)
    
    for cluster_index in range(k): 
        cur_x=[x[i] for i in range(num_points) if cluster_assignments[i]==cluster_index]
        cur_y=[y[i] for i in range(num_points) if cluster_assignments[i]==cluster_index]
        
        #handle the edge case of an empty cluster! 
        if len(cur_x)==0: 
            #reinitialize the centroid at 0,0 
            mean_x=0 
            mean_y=0 
        else:    
            mean_x=sum(cur_x)/len(cur_x)
            mean_y=sum(cur_y)/len(cur_y)
        
        new_x_centroids.append(mean_x)
        new_y_centroids.append(mean_y)
        
    return new_x_centroids,new_y_centroids





def scikit_PCAandkmeans(data,n_clusters,xlabel,ylabel,plottitle):
    from matplotlib import pyplot as plt 
    np.random.seed(1234) 
    reduced_data = PCA(n_components=2).fit_transform(data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    np.random.seed(1234) 
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(reduced_data)
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1],'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(plottitle+'\n Centroids are marked with white x')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
    cluster_labels=kmeans.fit_predict(reduced_data)
    silhouette_avg = silhouette_score(reduced_data, cluster_labels)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)

    return cluster_labels

def scikit_silhouette(data,n_clusters): 
    reduced_data = PCA(n_components=2).fit_transform(data)
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(reduced_data) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(reduced_data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(reduced_data, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

def plot_heatmap(data):
    import seaborn as sns;
    sns.set(color_codes=False)
    g = sns.clustermap(data)
    
    
def get_genes_from_clusters(data,clusters,k,filename):
    #create a dictionary mapping all differential gene id's to the corresponding gene names. 
    gene_id_to_gene_name=open(filename,'r').read().strip().split('\n')
    gene_id_to_gene_name_dict=dict()
    for line in gene_id_to_gene_name:
        tokens=line.split()
        gene_id_to_gene_name_dict[tokens[0].split('.')[0]]=tokens[1]
        
    for i in range(k):
        cur_cluster=np.where(clusters==i)
        cur_genes=data.index[cur_cluster]
        outf=open(str(i)+".txt",'w')
        for gene_id in cur_genes:
            try:
                gene_name=gene_id_to_gene_name_dict[gene_id.split('.')[0]]
                outf.write(gene_name+'\n')
            except:
                print(gene_id) 
                continue 
        outf.close() 
