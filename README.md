# TSP clustering heuristic
## Abstract
This is a implementation of the Travelers Salesman Problem (TSP) based on a real scenario in the city of Belo Horizonte. In this scnário we have 100 locations that need to be visited in a single circuit that starts and end in the base location. Because of the size this instance of the TSP can't be optimally solved, making the use of some heuristic necessary. In this work the structure of the demand locations is exploited by clustering it and reducing the problem to a TSP on the centroids of the clusters. The relative order of the cluster is fixed on the original problem so it can be solved.

## Data
To make this work close to a real scenario the demand locations are the locations of construction sites available for buying in the city of Belo Horizonte. The base location is the center of a nearby city (known for having many industries) and the objective is to in a single circuit visit all the demand locations e go back to the base.

## Model
A Mixed-Integer Programming (MIP) model based on the _Miller Tucker as Zemilin_ work was implemented in pyomo for solving the reduced and original problem. The distance matrix uses the cityblock distance and hierarchical clustering is used for reducing the problem. The number of cluters and method used can be set to diferent values

## Strucure
### Data
Folder with the data (coodinates of the locations)
### imgs
Maps of the solutions made using google static maps API. For using the make_solution_map() an API class method key is needed
### tsp.py
TSP class that implements the pyomo models and clustering algorithms
### Exemple.ipynb
Notebook with results obtained by using this method in the data
