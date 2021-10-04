import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib.lines import Line2D

def display_circles(X_projected, pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None, alpha=1, illustrative_var=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            # fig, ax = plt.subplots(figsize=(8,6))
            fig = plt.figure(figsize=(7,4))
  
            # affichage des points et centroïdes respectifs
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                valeurs = []
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    # if value == 1:
                        # plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='red') 
                        # plt.scatter((np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape)),(np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape)),
                        # alpha=alpha,label=value,cmap = 'Jet',marker='P',color='cyan',edgecolors='red',s=500,linewidth=3) 
                        # centroid_x_c1 = (np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape))
                        # centroid_y_c1 = (np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape))
                        # dataset_c1 = pd.DataFrame({'centroid_x_c1':centroid_x_c1, 'centroid_y_c1':centroid_y_c1},index=['cluster 1'])
                        # print(dataset_c1)
                        # valeurs.append([centroid_x_c1,centroid_y_c1])
                    # elif value == 2:
                        # plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='blue') 
                        # plt.scatter((np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape)),(np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape)),
                        # alpha=alpha,label=value,cmap = 'Jet',marker='P',color='cyan',edgecolors='blue',s=500,linewidth=3)   
                        # centroid_x_c2 = (np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape))
                        # centroid_y_c2 = (np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape))
                        # dataset_c2 = pd.DataFrame({'centroid_x_c2':centroid_x_c2, 'centroid_y_c2':centroid_y_c2},index=['cluster 2']) 
                        # print(dataset_c2)
                        # valeurs.append([centroid_x_c2,centroid_y_c2])
                    # elif value == 3:
                        # plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='midnightblue')
                        # plt.scatter((np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape)),(np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape)),
                        # alpha=alpha,label=value,cmap = 'Jet',marker='P',color='cyan',edgecolors='midnightblue',s=500,linewidth=3)  
                        # centroid_x_c3 = (np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape))
                        # centroid_y_c3 = (np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape))
                        # dataset_c3 = pd.DataFrame({'centroid_x_c3':centroid_x_c3, 'centroid_y_c3':centroid_y_c3},index=['cluster 3']) 
                        # print(dataset_c3)
                        # valeurs.append([centroid_x_c3,centroid_y_c3])
                    # elif value == 4:
                        # plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='green') 
                        # plt.scatter((np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape)),(np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape)),
                        # alpha=alpha,label=value,cmap = 'Jet',marker='P',color='cyan',edgecolors='green',s=500,linewidth=3)     
                        # centroid_x_c4 = (np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape))
                        # centroid_y_c4 = (np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape))
                        # dataset_c4 = pd.DataFrame({'centroid_x_c4':centroid_x_c4, 'centroid_y_c4':centroid_y_c4},index=['cluster 4']) 
                        # print(dataset_c4)
                        # valeurs.append([centroid_x_c4,centroid_y_c4])
                    # elif value == 5:
                        # plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='yellow')
                        # plt.scatter((np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape)),(np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape)),
                        # alpha=alpha,label=value,cmap = 'Jet',marker='P',color='cyan',edgecolors='yellow',s=500,linewidth=3)
                        # centroid_x_c5 = (np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape))
                        # centroid_y_c5 = (np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape))
                        # dataset_c5 = pd.DataFrame({'centroid_x_c5':centroid_x_c5, 'centroid_y_c5':centroid_y_c5},index=['cluster 5'])
                        # print(dataset_c5)
                        # valeurs.append([centroid_x_c5,centroid_y_c5])
                    # else :    
                        # plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='magenta')
                        # plt.scatter((np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape)),(np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape)),
                        # alpha=alpha,label=value,cmap = 'Jet',marker='P',color='cyan',edgecolors='magenta',s=500,linewidth=3)
                        # centroid_x_c6 = (np.sum(X_projected[selected, d1],axis=1)/np.prod((X_projected[selected, d1]).shape))
                        # centroid_y_c6 = (np.sum(X_projected[selected, d2],axis=1)/np.prod((X_projected[selected, d2]).shape))
                        # dataset_c6 = pd.DataFrame({'centroid_x_c6':centroid_x_c6, 'centroid_y_c6':centroid_y_c6},index=['cluster 6'])
                        # print(dataset_c6)
                        # valeurs.append([centroid_x_c6,centroid_y_c6])
                        # dataset_global = pd.DataFrame(valeurs,columns=['centroid_x', 'centroid_y'],index=['cluster 1','cluster 2','cluster 3','cluster 4','cluster 5','cluster 6'])
                        # dataset_global.to_csv("Documents/Dossier_AISSA/POLE_EMPLOI/OPENCLASSROOM/P6_mouacha_aissa/Liste_centroides.csv", index = True)
                
                #my_colors = {'cluster 1':'red', 'cluster 2':'blue', 'cluster 3':'midnightblue', 'cluster 4':'green', 'cluster 5':'yellow', 'cluster 6':'magenta'}
                # my_colors = {'C1':'blue','C2':'red'}
                # lab_col = list(my_colors.keys())
                # handles = [plt.Rectangle((0,0),0,0, color=my_colors[label]) for label in lab_col]
                # plt.legend(handles, lab_col, ncol=1, shadow=False, loc='upper left',fontsize=8)
                
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims #-boundary, boundary, -boundary, boundary #lims                                                           #2.5 factor to superpose with scatter
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1 #-boundary, boundary, -boundary, boundary #-1, 1, -1, 1                                                   #2.5 factor to superpose with scatter
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])# -boundary, boundary, -boundary, boundary #min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:]) #2.5 factor to superpose with scatter

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),pcs[d1,:], pcs[d2,:],angles='xy', scale_units='xy', scale=1, color="black") #2.5 factor to superpose with scatter
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='9', ha='left', va='center',rotation=label_rotation, color="black", alpha=1)  #2.5 factor to superpose with scatter
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='black') #1, facecolor='none', edgecolor='b')                                     #2.5 rayon factor to superpose with scatter
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-boundary, boundary], [0, 0], color='grey', ls='--')                                                                                      #2.5 rayon factor to superpose with scatter (old -1,1)
            plt.plot([0, 0], [-boundary, boundary], color='grey', ls='--')                                                                                      #2.5 rayon factor to superpose with scatter (old -1,1)

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            # plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,4))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    if value == 1:
                        plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='red') #color=illustrative_var.astype(np.float)
                    elif value == 2:
                        plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='blue') 
                    elif value == 3:
                        plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='midnightblue') 
                    elif value == 4:
                        plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='green') 
                    elif value == 5:
                        plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='yellow') 
                    else :
                        plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, cmap = 'Jet', color='magenta') 
                # plt.legend()
                #my_colors = {'cluster 1':'red', 'cluster 2':'blue', 'cluster 3':'midnightblue', 'cluster 4':'green', 'cluster 5':'yellow', 'cluster 6':'magenta'}
                my_colors = {'true':'blue','false':'red'}
                lab_col = list(my_colors.keys())
                handles = [plt.Rectangle((0,0),0,0, color=my_colors[label]) for label in lab_col]
                plt.legend(handles, lab_col, ncol=1, shadow=False, loc='upper left',fontsize=8)

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],fontsize='8', ha='left',va='top') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            # plt.show(block=False)

def display_scree_plot(pca):
    plt.figure(figsize=(7,4))
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    legend_elements = [
                        Line2D([0], [0], marker='o', color='red', label='Cumulative explained variance',markerfacecolor='red', markersize=6),
                        plt.Rectangle((0,0),0,0,     color='C0' , label='Individual explained variance')
                      ]
    plt.legend(handles=legend_elements, loc='upper left',shadow=True,fontsize=8)


def plot_dendrogram(Z, names):
    plt.figure(figsize=(12,10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
        color_threshold= 5.5
        
               )
    #fig = plt.gcf()
    #return fig         
    