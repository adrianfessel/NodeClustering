# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:16:16 2019

@author: Adrian
"""

import numpy as np
import networkx as nx
import pandas as pd
import scipy.spatial as spatial

import os
import cv2
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.svm import SVC

def r_neighbors(G, ID, r):
    """
    Function to obtain the IDs of all nodes that are separated by at most r 
    steps from the source node (measured along the shortest path). 
                                                                    

    Parameters
    ----------
    G : networkx graph
        undirected graph
    ID : int
        ID of the source node
    r : int
        maximum separation between source and target nodes

    Returns
    -------
    nbrs : list
        list of neighbor IDs

    """
    
    return [IDj for IDj in nx.single_source_shortest_path_length(G, ID, cutoff=r)]


def get_voronoi_points(x, y, X, Y, size):
    """
    Function to compute the voronoi tesselation for the set of points (x, y),
    sampled only at the points (Y, X).

    Parameters
    ----------
    x : numpy array
        x-coordinates centers
    y : numpy array
        y-coordinates centers
    X : numpy array
        x-coordinates samples
    Y : numpy array
        y-coordinates samples

    Returns
    -------
    V : numpy array
        sampled voronoi tesselation at points (X, Y)

    """
    
    pos = np.asarray([[xi, yi] for xi, yi in zip(x, y)])
    XY = np.c_[Y.ravel(), X.ravel()]
    
    Tree = spatial.cKDTree(pos)
    V = np.float32(Tree.query(XY)[1])

    return V


def sample_map(G, I):
    """
    Function to sample the gray values of an image in the regions covered by
    the radii of the nodes of an apollonian graph.

    Problem: extremely slow

    Parameters
    ----------
    G : networkx graph
        apollonian graph with fields 'r', 'x', 'y'
    I : numpy array
        grayscale image

    Returns
    -------
    samples : dict of lists
        [mean, std, min, max], keyed by node ID

    """
    
    X, Y = np.meshgrid(range(I.shape[1]),range(I.shape[0]))
    
    samples = {}
    
    for ID in G:
        mask = np.sqrt((G.nodes[ID]['x']-X)**2+(G.nodes[ID]['y']-Y)**2) <= G.nodes[ID]['r']
        gray = I[np.where(mask)]
        samples[ID] = [np.mean(gray), np.std(gray), np.min(gray), np.max(gray)]

    return samples


def collect_neighbor_features(ID, G, measures, r=0):
    """
    A function to compute a number of local features for a node. 
    Features are provided for the source node and for its neighbors of rank r.
    for each discrete value of r (r=0, 1, ...) all features are computed for
    the individual neighbors. Then, the mean, std and min/max statistics are
    computed for the set and attributed to the center node.

    Parameters
    ----------
    ID : int
        ID of the center node
    G : networkx graph
        apollonian graph with fields 'r', 'x', 'y'
    measures : dict
        dict of measures, keyed by ID
    r : int, optional
        max rank for r-neighbors; the default is 0, corresponding to only the
        source node.
    
    Returns
    -------
    features : dict
        dict of features, keyed by feature name

    """
        
    features = {}
    
    nbrs = {0:set([ID])}
    for r in range(1, r+1):
        nbrs[r] = set(r_neighbors(G, ID, r)) - set(r_neighbors(G, ID, r-1))

    methods = {'mean':np.mean, 'std':np.std, 'max':np.max, 'min':np.min}
    
    for r in nbrs:

        for measure_name, measure in measures.items():
            
            measure = [measure[i] for i in nbrs[r]]
            
            for method_name, method in methods.items():
                features[measure_name+method_name+str(r)] = method(measure) if len(measure) > 0 else np.nan
            
    return features


def features_single_frame(G, I, r=0):
    """
    Function for computing features for a single frame.

    Parameters
    ----------
    G : networkx graph
        apollonian graph with fields 'r', 'x', 'y'
    I : numpy array
        grayscale image
    r : int, optional
        max rank for r-neighbors; the default is 0, corresponding to only the
        source node

    Returns
    -------
    features : pandas dataframe
        dataframe containing features, keyed by node ID

    """
        
    degree = {i:G.degree(i) for i in G}
    radius = {i:G.nodes[i]['r'] for i in G}
    clustering = nx.clustering(G)
    centrality = nx.betweenness_centrality(G)
    gray = {i:I[G.nodes[i]['y'], G.nodes[i]['x']] for i in G}
    
    measures = {'gray' : gray,
                'degree' : degree,
                'radius' : radius,
                'clustering' : clustering,
                'centrality' : centrality
                }
    
    features = {}
    
    for ID in G:
        features[ID] = collect_neighbor_features(ID, G, measures=measures, r=r)
        
    return pd.DataFrame.from_dict(features, orient='index')

    
def cluster_single_frame(features, n=2):
    """
    Function for clustering a single frame.

    Parameters
    ----------
    features : pandas dataframe
        dataframe containing features, keyed by node ID
    n : int, optional
        number of clusters; the default is 2

    Returns
    -------
    labels : dict
        cluster assignment, keyed by node ID

    """
    
    pipeline = make_pipeline(KNNImputer(), StandardScaler(), KMeans(n_clusters=n))
    clustering = pipeline.fit_predict(features)
    
    labels = {ID:clustering[i] for i, ID in enumerate(features.index.values)}

    return labels


def cluster_multiple_frames(PathsG, PathsI, r=0, n=2):
    """
    Function to collect features for all nodes in a set of images/graphs. This
    function is intendend for improving clustering quality by being able to
    provide a larger set of training data.

    Parameters
    ----------
    PathsG : list of strings
        list of system paths to apollonian graphs
    PathsI : list of strings
        list of system paths to grayscale images
    r : int, optional
        max rank for r-neighbors; the default is 0, corresponding to only the
        source node
    n : int, optional
        number of clusters; the default is 2

    Returns
    -------
    features : pandas dataframe
        dataframe containing the features used for clustering
    labels : numpy array
        cluster assignment

    """
    
    # construct feature names
    
    features = {}
    for measure in ['radius', 'degree', 'radius', 'clustering', 'centrality']:
        for method in ['mean', 'std', 'min', 'max']:
            for ri in range(0, r+1):
                features[measure+method+str(ri)] = []
    
    # read data, compute features
    
    for PathG, PathI in zip(PathsG, PathsI):
    
        I = cv2.imread(PathI, cv2.IMREAD_GRAYSCALE)
        
        info = np.iinfo(I.dtype)
        I = np.float32(I) / info.max
        I = (I - I.mean()) / I.std()  
        
        G = pickle.load(open(PathG, 'rb'))
        
        degree = {i:G.degree(i) for i in G}
        radius = {i:G.nodes[i]['r'] for i in G}
        clustering = nx.clustering(G)
        centrality = nx.betweenness_centrality(G)
        gray = {i:I[G.nodes[i]['y'], G.nodes[i]['x']] for i in G}
        
        measures = {'gray' : gray,
                    'degree' : degree,
                    'radius' : radius,
                    'clustering' : clustering,
                    'centrality' : centrality
                    }
        
        for ID in G:
            
            node_features = collect_neighbor_features(ID, G, measures=measures, r=r)
            
            for name in features:
            
                features[name].append(node_features[name])
                
    # perform clustering
                
    features = pd.DataFrame.from_dict(features, orient='columns')

    pipeline = make_pipeline(KNNImputer(), StandardScaler(), KMeans(n_clusters=n))
    labels = pipeline.fit_predict(features)
    
    return features, labels


def sort_clusters_by_radius(features, labels):
    """
    Function to sort cluster indices by the average radius of nodes in the 
    cluster (descending order). Returns indices.

    Parameters
    ----------
    features : pandas dataframe
        dataframe containing the features used for clustering
    labels : numpy array
        cluster assignment

    Returns
    -------
    indices : list
        cluster indices, sorted by radius (descending order)

    """
    
    categories = np.unique(labels)
    radii = np.asarray(features['radiusmean0'])
    cluster_radius_mean = {}
    
    for category in categories:
        inds = np.where(labels == category)
        cluster_radius_mean[category] = np.mean(radii[inds])

    return sorted(cluster_radius_mean, key=cluster_radius_mean.get, reverse=True)


def svm_train(features, labels):
    """
    Train a support vector machine.

    Parameters
    ----------
    features : pandas dataframe
        
    labels : numpy array
        array of category labels

    Returns
    -------
    clf : classifier object
        trained svm classifier

    """

    clf = make_pipeline(KNNImputer(), StandardScaler(), SVC())    
    clf.fit(features, labels)
    
    return clf


def plot_graph_clustering(G, C):
    """
    Function for displaying the clustering result overlayed onto an apollonian
    graph.

    Parameters
    ----------
    G : networkx graph
        apollonian graph with fields 'r', 'x', 'y'
    C : dict
        dict of cluster associations, keyed by node ID

    Returns
    -------
    None.

    """

    nC = len(np.unique([C[ID] for ID in C]))

    Colors = plt.cm.jet(np.linspace(0,1,nC))

    for ID in G.nodes():
        c = plt.Circle((G.nodes[ID]['x'], G.nodes[ID]['y']), radius=G.nodes[ID]['r'], edgecolor=Colors[C[ID]], facecolor=Colors[C[ID]], alpha=0.25, linewidth=1)
        plt.gca().add_patch(c)
    
    pos = {ID:[G.nodes[ID]['x'],G.nodes[ID]['y']] for ID in G}
    nx.draw_networkx_edges(G, pos, width=1)

    plt.xlim([0, np.max([pos[i][0] for i in pos])])
    plt.ylim([0, np.max([pos[i][1] for i in pos])])
    
    plt.axis('equal')
    plt.axis('off')


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    Path = './'
    
    I = cv2.imread(os.path.join(Path, 'grayscale.png'), cv2.IMREAD_GRAYSCALE)
    
    info = np.iinfo(I.dtype)
    I = np.float32(I) / info.max
    I = (I - I.mean()) / I.std()  
    
    B = cv2.imread(os.path.join(Path, 'binary.png'), cv2.IMREAD_GRAYSCALE)
    G = pickle.load(open(os.path.join(Path, 'apollonian.pkl'), 'rb'))
    
    features = features_single_frame(G, I, r=3)
    C = cluster_single_frame(features, n=2)

    plt.figure()
    plt.imshow(I, cmap='gray')
    
    plot_graph_clustering(G, C)
