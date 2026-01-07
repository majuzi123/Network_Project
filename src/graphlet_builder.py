import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt


class GraphletConstructor:
    def __init__(self):
        self.host_graphlets = {}
        self.host_flows = defaultdict(list)

    def parse_network_flow(self, flow_string):
        """解析CSV格式流量字符串: srcIP,dstIP,protocol,sPort,dPort,label"""
        parts = flow_string.strip().split(',')

        if len(parts) >= 6:
            # CSV格式: srcIP,dstIP,protocol,sPort,dPort,label
            source_ip = int(parts[0])
            destination_ip = int(parts[1])  # CSV中第二个是dstIP
            protocol = int(parts[2])
            source_port = int(parts[3])
            destination_port = int(parts[4])

            # 处理标签（可能是normal/malicious或0/1）
            label_str = parts[5].strip().lower()
            if label_str == 'normal':
                label = 0
            elif label_str == 'malicious':
                label = 1
            else:
                try:
                    label = int(label_str)
                except:
                    label = None

            return source_ip, protocol, destination_ip, source_port, destination_port, label
        else:
            # 旧格式（空格分隔）保持向后兼容
            parts = flow_string.strip().split()
            if len(parts) >= 5:
                source_ip = int(parts[0])
                protocol = int(parts[1])
                destination_ip = int(parts[2])
                source_port = int(parts[3])
                destination_port = int(parts[4])
                label = int(parts[5]) if len(parts) > 5 else None

                return source_ip, protocol, destination_ip, source_port, destination_port, label

        raise ValueError(f"无法解析的流量字符串: {flow_string}")

    def add_flow_to_host(self, source_ip, protocol, destination_ip, source_port, destination_port, label=None):
        if source_ip not in self.host_graphlets:
            self.host_graphlets[source_ip] = nx.DiGraph()
            self.host_flows[source_ip] = []

        flow_data = {
            'source_ip': source_ip,
            'protocol': protocol,
            'destination_ip': destination_ip,
            'source_port': source_port,
            'destination_port': destination_port,
            'label': label
        }
        self.host_flows[source_ip].append(flow_data)

        graphlet_nodes = [
            f"source_ip_{source_ip}",
            f"protocol_{protocol}",
            f"destination_ip_{destination_ip}",
            f"source_port_{source_port}",
            f"destination_port_{destination_port}",
            f"destination_ip2_{destination_ip}"
        ]

        for node in graphlet_nodes:
            node_type = node.split('_')[0]
            node_value = node.split('_')[1]
            self.host_graphlets[source_ip].add_node(node, node_type=node_type, node_value=node_value)

        for i in range(len(graphlet_nodes) - 1):
            self.host_graphlets[source_ip].add_edge(graphlet_nodes[i], graphlet_nodes[i + 1])

    def load_csv_data(self, filename, has_labels=True):
        """从CSV文件加载流量数据"""
        print(f"Loading CSV data from {filename}")
        with open(filename, 'r') as file:
            lines = file.readlines()

        all_labels = []
        for line_number, line_content in enumerate(lines):
            try:
                src, proto, dst, sport, dport, lbl = self.parse_network_flow(line_content)
                if not has_labels:
                    lbl = None
                self.add_flow_to_host(src, proto, dst, sport, dport, lbl)
                if lbl is not None:
                    all_labels.append(lbl)
            except Exception as e:
                print(f"Error parsing line {line_number + 1}: {line_content}")
                print(f"Error details: {e}")

        print(f"Loaded {len(self.host_graphlets)} unique hosts from CSV")
        if all_labels:
            print(f"Normal flows: {all_labels.count(0)}, Malicious flows: {all_labels.count(1)}")
        return all_labels

    def extract_important_nodes(self, host_id):
        if host_id not in self.host_graphlets:
            return []

        host_graph = self.host_graphlets[host_id]
        important_nodes = []

        for node in host_graph.nodes():
            incoming_edges = host_graph.in_degree(node)
            outgoing_edges = host_graph.out_degree(node)
            if incoming_edges > 1 or outgoing_edges > 1:
                important_nodes.append(node)

        return important_nodes

    def create_profile_graphlet(self, host_id):
        if host_id not in self.host_graphlets:
            return nx.DiGraph()

        original_graph = self.host_graphlets[host_id]
        important_nodes_set = set(self.extract_important_nodes(host_id))

        profile_graph = nx.DiGraph()

        for node in important_nodes_set:
            node_attributes = original_graph.nodes[node]
            profile_graph.add_node(node, **node_attributes)

        for edge_start, edge_end in original_graph.edges():
            if edge_start in important_nodes_set and edge_end in important_nodes_set:
                profile_graph.add_edge(edge_start, edge_end)

        return profile_graph

    def display_graphlet(self, host_id, show_profile=False):
        if host_id not in self.host_graphlets:
            print(f"Host {host_id} not found")
            return

        if show_profile:
            display_graph = self.create_profile_graphlet(host_id)
            graph_title = f"Profile Graphlet for Host {host_id}"
        else:
            display_graph = self.host_graphlets[host_id]
            graph_title = f"Activity Graphlet for Host {host_id}"

        plt.figure(figsize=(12, 8))
        positions = nx.spring_layout(display_graph, seed=42)

        node_colors_list = []
        for node in display_graph.nodes():
            node_type = display_graph.nodes[node]['node_type']
            if 'source' in node_type:
                node_colors_list.append('lightblue')
            elif 'protocol' in node_type:
                node_colors_list.append('lightgreen')
            elif 'destination' in node_type:
                node_colors_list.append('lightcoral')
            elif 'port' in node_type:
                node_colors_list.append('gold')
            else:
                node_colors_list.append('gray')

        nx.draw(display_graph, positions, with_labels=True, node_color=node_colors_list,
                node_size=800, font_size=8, edge_color='gray', arrowsize=10, width=1.5)

        plt.title(graph_title)
        plt.show()

        print(f"Node count: {display_graph.number_of_nodes()}")
        print(f"Edge count: {display_graph.number_of_edges()}")
        print(f"Important nodes: {len(self.extract_important_nodes(host_id))}")