import numpy as np
import networkx as nx


class WalkKernelExtractor:
    def __init__(self, walk_steps=4):
        self.walk_steps = walk_steps
        self.feature_mapping = {}
        self.feature_counter = 0

    def generate_walks(self, graph_structure, max_walk_count=1000):
        walks_collection = []
        all_nodes = list(graph_structure.nodes())

        start_nodes = all_nodes[:min(50, len(all_nodes))]

        for start_node in start_nodes:
            current_walks = self._perform_random_walk(graph_structure, start_node)
            walks_collection.extend(current_walks)

            if len(walks_collection) >= max_walk_count:
                walks_collection = walks_collection[:max_walk_count]
                break

        return walks_collection

    def _perform_random_walk(self, graph_structure, start_node, walks_per_node=20):
        walks_result = []

        for _ in range(walks_per_node):
            walk_path = [start_node]
            current_position = start_node

            for step in range(self.walk_steps):
                neighbor_nodes = list(graph_structure.successors(current_position))

                if not neighbor_nodes:
                    break

                next_node = np.random.choice(neighbor_nodes)
                walk_path.append(next_node)
                current_position = next_node

            if len(walk_path) == self.walk_steps + 1:
                walks_result.append(tuple(walk_path))

        return walks_result

    def convert_walks_to_features(self, walks_collection, use_hashing=True, feature_size=1024):
        if use_hashing:
            feature_vector = np.zeros(feature_size)

            for walk in walks_collection:
                walk_string = '-'.join([str(hash(node)) for node in walk])
                feature_index = hash(walk_string) % feature_size
                feature_vector[feature_index] += 1

            return feature_vector
        else:
            feature_dict = {}

            for walk in walks_collection:
                walk_string = '-'.join([str(node) for node in walk])
                if walk_string not in self.feature_mapping:
                    self.feature_mapping[walk_string] = self.feature_counter
                    self.feature_counter += 1

                idx = self.feature_mapping[walk_string]
                feature_dict[idx] = feature_dict.get(idx, 0) + 1

            dense_vector = np.zeros(self.feature_counter)
            for idx, count in feature_dict.items():
                dense_vector[idx] = count

            return dense_vector

    def compute_kernel_matrix(self, graphlets_list):
        graph_count = len(graphlets_list)
        kernel_matrix = np.zeros((graph_count, graph_count))

        all_walks_list = []
        for graph in graphlets_list:
            walks = self.generate_walks(graph, max_walk_count=500)
            all_walks_list.append(set(walks))

        for i in range(graph_count):
            for j in range(i, graph_count):
                common_walks = len(all_walks_list[i].intersection(all_walks_list[j]))
                kernel_matrix[i][j] = common_walks
                kernel_matrix[j][i] = common_walks

        return kernel_matrix


class AdvancedWalkKernel(WalkKernelExtractor):
    def generate_walks(self, graph_structure, max_walk_count=1000):
        walks_collection = []
        all_nodes = list(graph_structure.nodes())

        start_nodes = all_nodes[:min(50, len(all_nodes))]

        for start_node in start_nodes:
            current_walks = self._advanced_walk(graph_structure, start_node)
            walks_collection.extend(current_walks)

            if len(walks_collection) >= max_walk_count:
                walks_collection = walks_collection[:max_walk_count]
                break

        return walks_collection

    def _advanced_walk(self, graph_structure, start_node, walks_per_node=20):
        walks_result = []

        for _ in range(walks_per_node):
            walk_path = [start_node]
            current_position = start_node
            previous_node = None

            for step in range(self.walk_steps):
                neighbor_nodes = list(graph_structure.successors(current_position))

                if not neighbor_nodes:
                    break

                if previous_node is not None and len(neighbor_nodes) > 1:
                    neighbor_nodes = [n for n in neighbor_nodes if n != previous_node]

                if not neighbor_nodes:
                    neighbor_nodes = list(graph_structure.successors(current_position))

                next_node = np.random.choice(neighbor_nodes)
                walk_path.append(next_node)
                previous_node = current_position
                current_position = next_node

            if len(walk_path) == self.walk_steps + 1:
                walks_result.append(tuple(walk_path))

        return walks_result