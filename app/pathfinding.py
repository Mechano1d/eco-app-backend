import random
import networkx as nx
import heapq
import math


class pathfinding:

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def get_weight(self, end_1, end_2, attr=None):
        data = self.analyzer.pollution_data.set_index("node_id")
        try:
            start_node = data.loc[end_1]
            end_node = data.loc[end_2]
            weight = (int(start_node['ow_aqi']) + int(end_node['ow_aqi'])) / 2
            return weight
        except KeyError as e:
            print(f"Помилка: не знайдено вузол {e}")
            return float('inf')

    def euclidean_distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def pathfind_star(self, start, finish):
        G = self.analyzer.graph
        data = self.analyzer.pollution_data.set_index("node_id")

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {node: float("inf") for node in G.nodes}
        g_score[start] = 0

        f_score = {node: float("inf") for node in G.nodes}
        start_pos = (data.loc[start]["latitude"], data.loc[start]["longitude"])
        finish_pos = (data.loc[finish]["latitude"], data.loc[finish]["longitude"])
        f_score[start] = self.euclidean_distance(start_pos, finish_pos)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == finish:
                # Відновлення шляху
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in G.neighbors(current):
                tentative_g = g_score[current] + self.get_weight(current, neighbor)
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    neighbor_pos = (data.loc[neighbor]["latitude"], data.loc[neighbor]["longitude"])
                    f_score[neighbor] = g_score[neighbor] + self.euclidean_distance(neighbor_pos, finish_pos)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # Якщо шляху не знайдено

    def pathfind(self, start, finish):
        G = self.analyzer.graph
        path = nx.shortest_path(G, source=start, target=finish, weight=self.get_weight)
        return path
