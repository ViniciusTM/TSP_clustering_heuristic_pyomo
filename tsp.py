import pandas as pd
import numpy as np
from collections import defaultdict
import pyomo.opt
import pyomo.environ as pe
from scipy.cluster.hierarchy import linkage, fcluster

colors = ["0x000000", "0x1CE6FF", "0xFFFF00", "0xFF34FF", "0xFF4A46", "0x008941", "0x006FA6", "0xA30059",
          "0xFFDBE5", "0x7A4900", "0x0000A6", "0x63FFAC", "0xB79762", "0x004D43", "0x8FB0FF", "0x997D87",
          "0x5A0007", "0x809693", "0xFEFFE6", "0x1B4400", "0x4FC601", "0x3B5DFF", "0x4A3B53", "0xFF2F80",
          "0x61615A", "0xBA0900", "0x6B7900", "0x00C2A0", "0xFFAA92", "0xFF90C9", "0xB903AA", "0xD16100",
          "0xDDEFFF", "0x000035", "0x7B4F4B", "0xA1C299", "0x300018", "0x0AA6D8", "0x013349", "0x00846F",
          "0x372101", "0xFFB500", "0xC2FFED", "0xA079BF", "0xCC0744", "0xC0B9B2", "0xC2FF99", "0x001E09",
          "0x00489C", "0x6F0062", "0x0CBD66", "0xEEC3FF", "0x456D75", "0xB77B68", "0x7A87A1", "0x788D66",
          "0x885578", "0xFAD09F", "0xFF8A9A", "0xD157A0", "0xBEC459", "0x456648", "0x0086ED", "0x886F4C",
          "0x34362D", "0xB4A8BD", "0x00A6AA", "0x452C2C", "0x636375", "0xA3C8C9", "0xFF913F", "0x938A81",
          "0x575329", "0x00FECF", "0xB05B6F", "0x8CD0FF", "0x3B9700", "0x04F757", "0xC8A1A1", "0x1E6E00",
          "0x7900D7", "0xA77500", "0x6367A9", "0xA05837", "0x6B002C", "0x772600", "0xD790FF", "0x9B9700",
          "0x549E79", "0xFFF69F", "0x201625", "0x72418F", "0xBC23FF", "0x99ADC0", "0x3A2465", "0x922329",
          "0x5B4534", "0xFDE8DC", "0x404E55", "0x0089A3", "0xCB7E98", "0xA4E804", "0x324E72", "0x6A3A4C",
          "0x83AB58", "0x001C1E", "0xD1F7CE", "0x004B28", "0xC8D0F6", "0xA3A489", "0x806C66", "0x222800",
          "0xBF5650", "0xE83000", "0x66796D", "0xDA007C", "0xFF1A59", "0x8ADBB4", "0x1E0200", "0x5B4E51",
          "0xC895C5", "0x320033", "0xFF6832", "0x66E1D3", "0xCFCDAC", "0xD0AC94", "0x7ED379", "0x012C58"]
google_key = GOOGLE_KEY

class TSP():
    def __init__(self, data):
        self.data = data
        self.data['cluster'] = np.repeat(1,len(data))
        self.data.at[0,'cluster'] = 0
        self.num_clusters = 2

    def solve_with_heuristic(self, num_clusters, method='ward', max_time=1800):
        summary = {}
        opt = pyomo.opt.SolverFactory('glpk')
        opt.options['tmlim'] = max_time
        self.cluster = self.make_clusters(num_clusters, method)

        # Creating and solving reduced problem
        reduced_distances = {(i,j):self.calc_dist(x,y) for i,x in self.clusters.iterrows() for j,y in self.clusters.iterrows()}
        reduced_model = self.create_model(reduced_distances, num_clusters)
        results = opt.solve(reduced_model)
        try:
             obj_func = pe.value(reduced_model.obj)
        except ValueError:
            print("Solution for the reduced problem not found")
            return None

        # Creating original problem
        distances = {(i,j):self.calc_dist(x,y) for i,x in self.data.iterrows() for j,y in self.data.iterrows()}
        model = self.create_model(distances, len(self.data)-1)

        # Fixing clusters relative positions
        for clust_idx in reduced_distances.keys():
            if clust_idx[0] != clust_idx[1] and pe.value(reduced_model.x[clust_idx]) == 0:
                idx1 = self.data[self.data['cluster'] == clust_idx[0]].index
                idx2 = self.data[self.data['cluster'] == clust_idx[1]].index
                fix_idx = [edge for edge in distances.keys() if (edge[0] in idx1) and (edge[1] in idx2)]
                for idx in fix_idx:
                    model.x[idx].fix(0)

        # Solving original problem
        results = opt.solve(model)
        try:
             obj_func = pe.value(model.obj)
        except ValueError:
            print("Solution for Original problem not found")
            return None

        self.solution = []
        for idx in distances.keys():
            if pe.value(model.x[idx]) == 1:
                self.solution.append(idx)

        return obj_func

    def make_clusters(self, num_clusters, method):
        if method == 'ward':
            metric='euclidean'
        else:
            metric='cityblock'

        x = self.data.iloc[1:][['lat', 'lng']].values
        lk = linkage(x, method=method, metric=metric)
        self.data['cluster'] = [0] + list(fcluster(lk, num_clusters, 'maxclust'))
        self.clusters = self.data.groupby('cluster').mean()[['lat', 'lng']]
        self.clusters['proc_times'] = self.data.groupby('cluster').sum()['proc_times']
        self.num_clusters = num_clusters + 1

    def make_map(self):
        base_url = 'https://maps.googleapis.com/maps/api/staticmap?size=800x500&scale=1' + '&key=' + google_key
        for c in range(self.num_clusters+1):
            base_url += '&markers=size:tiny|color:' + colors[c]
            for _,row in self.data[self.data['cluster'] == c].iterrows():
                base_url += '|' + '{:f},{:f}'.format(row['lat'], row['lng'])

        return base_url

    def make_solution_map(self):
        try:
            self.solution
        except AttributteError:
            print('No solution was found yet')
            return None

        route = [0]
        for i in range(len(self.solution)):
            route.append(next(i[1] for i in self.solution if i[0]==route[-1]))

        base_url = self.make_map()
        base_url += '|&path=color:0x999966'
        for loc in route:
            row = self.data.iloc[loc]
            base_url += '|' + '{:f},{:f}'.format(row['lat'],row['lng'])

        return base_url

    @staticmethod
    def solve_model(model):
        opt = pyomo.opt.SolverFactory('glpk')
        result = opt.solve(model)

        x_index = [(i,j) for i in model.N for j in model.N]
        solution = []
        for idx in x_index:
            if pe.value(model.x[idx]) == 1:
                solution.append((idx[0],idx[1]))


        return solution




        return solution

    @staticmethod
    def calc_dist(x,y):
        delta_lat = abs(x['lat']-y['lat'])*0.01745329252
        delta_lng = abs(x['lng']-y['lng'])*0.01745329252

        a = (np.sin(delta_lat/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        latDist = 6371 * c

        a = (np.sin(delta_lng/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        lngDist = 6371 * c

        return abs(latDist) + abs(lngDist)

    @staticmethod
    def create_model(distances, n):
        model = pe.ConcreteModel()
        # Sets
        model.N = pe.Set(initialize = range(n+1))  # Locations
        model.N_ = pe.Set(initialize = range(1,n+1)) # Locations without origin

        # Parameters
        model.d = pe.Param(model.N, model.N, initialize=distances) # Distance Matrix

        # Varibles
        # Decision variable 1 -> if link (i,j) is on the answer
        model.x = pe.Var(model.N, model.N, domain=pe.Binary)
        # Circuit eliminating varible
        model.u = pe.Var(model.N_, domain=pe.NonNegativeReals)

        # Objective Function
        def obj_rule(model):
            return sum([model.x[i,j]*model.d[i,j] for i in model.N for j in model.N])
        model.obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

        # Restricao 1
        def r1_rule(model, i):
            return sum(model.x[i,j] for j in model.N) == 1
        model.r1 = pe.Constraint(model.N_, rule=r1_rule)

        # Restricao 2
        def r2_rule(model, j):
            return sum(model.x[i,j] for i in model.N) == 1
        model.r2 = pe.Constraint(model.N_, rule=r2_rule)

        # Restricao 3
        def r3_rule(model, h):
            return sum(model.x[i,h] for i in model.N) - sum(model.x[h,j] for j in model.N) == 0
        model.r3 = pe.Constraint(model.N, rule=r3_rule)

        # Restricao 4
        def r4_rule(model, i, j):
            return model.u[i] - model.u[j] + n*model.x[i,j] <= n-1
        model.r4 = pe.Constraint(model.N_, model.N_, rule=r4_rule)

        return model
