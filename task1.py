import heapq
import networkx as nx
import matplotlib.pyplot as plt

class Dijkstra:
    def __init__(self, graph: dict[tuple[str, int]]):
        self.graph = graph

    def find_path(self, start):
        distances = {v: float('inf') for v in self.graph}
        distances[start] = 0
        predecessors = {v: None for v in self.graph}
        pq = [(0, start)]

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)

            for neighbor, weight in self.graph[current_vertex]:
                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(pq, (new_distance, neighbor))

        return distances, predecessors

    def reconstruct_path(self, start, end, predecessors):
        if start is None or end is None:
            return "Invalid start or end vertex"
        path = []
        while end is not None:
            path.append(end)
            end = predecessors[end]
        return path[::-1]

    def visualize(self, path=None, path_color='lightgreen', graph_color='lightblue'):
        G = nx.Graph()
        for node, neighbors in self.graph.items():
            G.add_weighted_edges_from((node, neighbor, weight) for neighbor, weight in neighbors)

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color=graph_color, edge_color='black')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=path_color, node_size=700)
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=path_color, width=2)

        plt.show()



graph_data = {
    'A': [('B', 2), ('C', 4)],
    'B': [('A', 2), ('C', 1), ('D', 7)],
    'C': [('A', 4), ('B', 1), ('E', 3), ('G', 6)],
    'D': [('B', 7), ('F', 2)],
    'E': [('C', 3), ('F', 5)],
    'F': [('D', 2), ('E', 5), ('G', 1)],
    'G': [('C', 6), ('F', 1), ('H', 3)],
    'H': [('G', 3)]
}

dijkstra = Dijkstra(graph_data)
start_vertex = 'A'
distances, predecessors = dijkstra.find_path(start_vertex)

print(f"Shortest paths from {start_vertex}:")
for vertex, distance in distances.items():
    path = dijkstra.reconstruct_path(start_vertex, vertex, predecessors)
    print(f"{start_vertex} -> {vertex}: {distance}, Path: {' -> '.join(path)}")
    dijkstra.visualize(path=path)
