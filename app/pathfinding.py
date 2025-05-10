import random

import networkx as nx


class pathfinding:

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def get_weight(self, end_1, end_2, attr):
        start_node = self.analyzer.pollution_data.loc[(self.analyzer.pollution_data['node_id'] == end_1)]
        end_node = self.analyzer.pollution_data.loc[(self.analyzer.pollution_data['node_id'] == end_2)]
        weight = int(start_node['ow_aqi'].iloc[0]) + int(end_node['ow_aqi'].iloc[0]) / 2
        return weight

    def pathfind(self, start, finish):
        G = self.analyzer.graph
        path = nx.shortest_path(G, source=start, target=finish, weight=self.get_weight)
        return path
