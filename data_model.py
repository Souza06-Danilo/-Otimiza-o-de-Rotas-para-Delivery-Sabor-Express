# data_model.py

from typing import List, Tuple, Dict, Optional
import math

class Node:
    """Representa um nó no grafo (local de entrega ou ponto de interesse)."""
    def __init__(self, id: int, name: str, lat: float, lon: float):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return f"Node(id={self.id}, name='{self.name}', coords=({self.lat:.4f}, {self.lon:.4f}))"

class Edge:
    """Representa uma aresta no grafo (rua ou conexão entre locais)."""
    def __init__(self, source_id: int, target_id: int, distance: float, time: float):
        self.source_id = source_id
        self.target_id = target_id
        self.distance = distance  # Peso 1: Distância em km
        self.time = time          # Peso 2: Tempo estimado de viagem em minutos

    def __repr__(self):
        return f"Edge(from={self.source_id} to={self.target_id}, dist={self.distance:.2f}km, time={self.time:.2f}min)"

class Graph:
    """Representação do grafo da cidade."""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.adj: Dict[int, List[Tuple[int, Edge]]] = {} # {source_id: [(target_id, edge)]}

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        if node.id not in self.adj:
            self.adj[node.id] = []

    def add_edge(self, edge: Edge, bidirectional: bool = True):
        # Adiciona a aresta de ida
        if edge.source_id in self.adj and edge.target_id in self.nodes:
            self.adj[edge.source_id].append((edge.target_id, edge))
        
        # Adiciona a aresta de volta (se for bidirecional)
        if bidirectional and edge.target_id in self.adj and edge.source_id in self.nodes:
            # Cria uma aresta de volta com as mesmas propriedades (simplificação)
            reverse_edge = Edge(edge.target_id, edge.source_id, edge.distance, edge.time)
            self.adj[edge.target_id].append((edge.source_id, reverse_edge))

class Order:
    """Representa um pedido de entrega."""
    def __init__(self, id: int, node_id: int, volume: float = 1.0, weight: float = 0.5):
        self.id = id
        self.node_id = node_id  # ID do nó de destino no grafo
        self.volume = volume    # Volume do pedido (para restrições de capacidade)
        self.weight = weight    # Peso do pedido (para restrições de capacidade)
        self.cluster_id: Optional[int] = None # ID do cluster após o K-Means

    def __repr__(self):
        return f"Order(id={self.id}, node_id={self.node_id}, cluster={self.cluster_id})"

class Courier:
    """Representa um entregador."""
    def __init__(self, id: int, name: str, capacity_volume: float = 10.0, capacity_weight: float = 5.0):
        self.id = id
        self.name = name
        self.capacity_volume = capacity_volume
        self.capacity_weight = capacity_weight
        self.current_location_node_id: int = 0 # Assume-se que o nó 0 é o restaurante/depósito

    def __repr__(self):
        return f"Courier(id={self.id}, name='{self.name}', capacity_vol={self.capacity_volume}, loc={self.current_location_node_id})"

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância Haversine entre dois pontos em km."""
    R = 6371  # Raio da Terra em km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def estimate_time(distance_km: float, avg_speed_kph: float = 20.0) -> float:
    """Estima o tempo de viagem em minutos."""
    # Tempo = Distância / Velocidade * 60 (para converter horas em minutos)
    return (distance_km / avg_speed_kph) * 60

# --- Dados de Exemplo ---

def create_example_data() -> Tuple[Graph, List[Order], List[Courier]]:
    """Cria um grafo de exemplo, pedidos e entregadores."""
    graph = Graph()
    
    # 1. Nós (Locais de Entrega/Pontos de Interesse)
    # Coordenadas fictícias para uma área central
    nodes_data = [
        (0, "Restaurante Sabor Express", -23.5505, -46.6333), # Depósito/Restaurante
        (1, "Rua A, 100", -23.5480, -46.6300),
        (2, "Av. B, 50", -23.5520, -46.6350),
        (3, "Praça C", -23.5550, -46.6380),
        (4, "Rua D, 200", -23.5450, -46.6350),
        (5, "Av. E, 300", -23.5580, -46.6310),
        (6, "Rua F, 10", -23.5400, -46.6400),
        (7, "Rua G, 500", -23.5600, -46.6360),
        (8, "Av. H, 150", -23.5530, -46.6420),
        (9, "Rua I, 75", -23.5420, -46.6320),
        (10, "Ponto J", -23.5570, -46.6400),
    ]
    
    for id, name, lat, lon in nodes_data:
        graph.add_node(Node(id, name, lat, lon))

    # 2. Arestas (Conexões entre os Nós)
    # Conexões baseadas em proximidade, com distâncias e tempos estimados
    # A velocidade média é de 20 km/h (333.33 m/min)
    
    # Função auxiliar para criar e adicionar arestas
    def add_calculated_edge(g: Graph, id1: int, id2: int, avg_speed: float = 20.0):
        node1 = g.nodes[id1]
        node2 = g.nodes[id2]
        dist = haversine_distance(node1.lat, node1.lon, node2.lat, node2.lon)
        time = estimate_time(dist, avg_speed)
        g.add_edge(Edge(id1, id2, dist, time))

    # Conexões do Restaurante (Nó 0)
    add_calculated_edge(graph, 0, 1)
    add_calculated_edge(graph, 0, 2)
    add_calculated_edge(graph, 0, 4)
    add_calculated_edge(graph, 0, 9)

    # Outras conexões
    add_calculated_edge(graph, 1, 4)
    add_calculated_edge(graph, 1, 9)
    add_calculated_edge(graph, 2, 3)
    add_calculated_edge(graph, 2, 8)
    add_calculated_edge(graph, 3, 5)
    add_calculated_edge(graph, 3, 7)
    add_calculated_edge(graph, 4, 6)
    add_calculated_edge(graph, 5, 7)
    add_calculated_edge(graph, 6, 8)
    add_calculated_edge(graph, 7, 10)
    add_calculated_edge(graph, 8, 10)
    add_calculated_edge(graph, 9, 6)
    
    # 3. Pedidos (Entregas a serem feitas)
    # Usaremos 15 pedidos para simular um horário de pico
    orders_data = [
        # Cluster 1 (Norte/Leste)
        (101, 1), (102, 4), (103, 9), (104, 1), (105, 4),
        # Cluster 2 (Sul/Oeste)
        (106, 3), (107, 7), (108, 5), (109, 7), (110, 3),
        # Cluster 3 (Centro/Oeste)
        (111, 2), (112, 8), (113, 10), (114, 8), (115, 2),
    ]
    
    orders = [Order(id, node_id) for id, node_id in orders_data]
    
    # 4. Entregadores
    couriers = [
        Courier(1, "João (Moto)", capacity_volume=15.0, capacity_weight=7.0),
        Courier(2, "Maria (Bicicleta)", capacity_volume=5.0, capacity_weight=3.0),
        Courier(3, "Pedro (Moto)", capacity_volume=15.0, capacity_weight=7.0),
    ]
    
    return graph, orders, couriers

if __name__ == '__main__':
    graph, orders, couriers = create_example_data()
    
    print("--- Grafo de Exemplo ---")
    print(f"Total de Nós: {len(graph.nodes)}")
    print(f"Total de Arestas (bidirecionais): {sum(len(v) for v in graph.adj) // 2}")
    print("\nNós:")
    for node_id, node in graph.nodes.items():
        print(node)
        
    print("\nPedidos:")
    for order in orders:
        print(order)
        
    print("\nEntregadores:")
    for courier in couriers:
        print(courier)
        
    # Exemplo de cálculo de distância
    node0 = graph.nodes[0]
    node1 = graph.nodes[1]
    dist_0_1 = haversine_distance(node0.lat, node0.lon, node1.lat, node1.lon)
    time_0_1 = estimate_time(dist_0_1)
    print(f"\nDistância Haversine (0 para 1): {dist_0_1:.4f} km")
    print(f"Tempo Estimado (0 para 1): {time_0_1:.2f} minutos")
