# algorithms.py

import numpy as np
from typing import List, Dict, Tuple, Optional
from data_model import Graph, Order, Node, Courier, haversine_distance
import heapq

# --- Algoritmo de Clustering (K-Means) ---

class KMeansClustering:
    """Implementação simplificada do K-Means para agrupar pedidos por localização."""
    
    def __init__(self, n_clusters: int, max_iter: int = 100, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids: np.ndarray = np.array([])
        
    def _initialize_centroids(self, data: np.ndarray):
        """Inicializa os centróides aleatoriamente a partir dos pontos de dados."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Seleciona n_clusters índices únicos aleatórios
        indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        self.centroids = data[indices]
        
    def _assign_clusters(self, data: np.ndarray) -> np.ndarray:
        """Atribui cada ponto de dados ao centróide mais próximo."""
        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
        
    def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recalcula os centróides como a média dos pontos em cada cluster."""
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.n_clusters):
            points = data[labels == i]
            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)
            else:
                # Se um cluster ficar vazio, reinicializa o centróide
                new_centroids[i] = self.centroids[i] 
        return new_centroids
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """Executa o algoritmo K-Means e retorna os rótulos dos clusters."""
        if self.n_clusters > data.shape[0]:
            raise ValueError("O número de clusters não pode ser maior que o número de pontos de dados.")
            
        self._initialize_centroids(data)
        
        labels = np.zeros(data.shape[0])
        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            labels = self._assign_clusters(data)
            self.centroids = self._update_centroids(data, labels)
            
            # Critério de parada: se os centróides não mudarem significativamente
            if np.allclose(old_centroids, self.centroids):
                break
                
        return labels

# --- Algoritmo de Busca (A*) ---

class AStarPathfinder:
    """Implementação do algoritmo A* para encontrar o caminho mais curto no grafo."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        
    def _heuristic(self, node_id_a: int, node_id_b: int) -> float:
        """Heurística: Distância Haversine (distância em linha reta) entre os nós."""
        node_a = self.graph.nodes[node_id_a]
        node_b = self.graph.nodes[node_id_b]
        # A distância Haversine é uma heurística admissível (nunca superestima o custo real)
        return haversine_distance(node_a.lat, node_a.lon, node_b.lat, node_b.lon)
        
    def find_shortest_path(self, start_node_id: int, end_node_id: int, cost_type: str = 'time') -> Optional[Tuple[List[int], float]]:
        """
        Encontra o caminho mais curto entre dois nós usando A*.
        
        Args:
            start_node_id: ID do nó de partida.
            end_node_id: ID do nó de destino.
            cost_type: Tipo de custo a ser minimizado ('time' ou 'distance').
            
        Returns:
            Uma tupla contendo a lista de IDs dos nós no caminho e o custo total,
            ou None se nenhum caminho for encontrado.
        """
        if start_node_id not in self.graph.nodes or end_node_id not in self.graph.nodes:
            return None

        # Fila de prioridade: (f_score, node_id)
        # f_score = g_score + h_score
        # g_score: custo do caminho do início até o nó atual
        # h_score: custo estimado do nó atual até o destino (heurística)
        open_set = [(0, start_node_id)]
        
        # g_score: custo do caminho mais barato do início até o nó
        g_score: Dict[int, float] = {node_id: float('inf') for node_id in self.graph.nodes}
        g_score[start_node_id] = 0
        
        # came_from: armazena o predecessor de cada nó no caminho mais curto
        came_from: Dict[int, Optional[int]] = {node_id: None for node_id in self.graph.nodes}
        
        while open_set:
            # Pega o nó com o menor f_score
            current_f_score, current_node_id = heapq.heappop(open_set)
            
            if current_node_id == end_node_id:
                # Caminho encontrado, reconstrói o caminho
                path = []
                current = end_node_id
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, g_score[end_node_id]
            
            # Explora os vizinhos
            for neighbor_id, edge in self.graph.adj.get(current_node_id, []):
                # Determina o custo da aresta
                if cost_type == 'time':
                    edge_cost = edge.time
                elif cost_type == 'distance':
                    edge_cost = edge.distance
                else:
                    raise ValueError("cost_type deve ser 'time' ou 'distance'")
                    
                # Custo do caminho do início até o vizinho através do nó atual
                tentative_g_score = g_score[current_node_id] + edge_cost
                
                if tentative_g_score < g_score[neighbor_id]:
                    # Este é um caminho melhor
                    came_from[neighbor_id] = current_node_id
                    g_score[neighbor_id] = tentative_g_score
                    
                    # Calcula o f_score para o vizinho
                    h_score = self._heuristic(neighbor_id, end_node_id)
                    f_score = tentative_g_score + h_score
                    
                    # Adiciona ou atualiza o vizinho na fila de prioridade
                    heapq.heappush(open_set, (f_score, neighbor_id))
                    
        return None # Nenhum caminho encontrado

# --- Otimização de Rota (VRP Simplificado) ---

class RouteOptimizer:
    """
    Classe para orquestrar o agrupamento e a otimização de rotas.
    
    Para o problema de otimização de rota (Vehicle Routing Problem - VRP)
    dentro de cada cluster, usaremos uma abordagem simplificada, como o
    algoritmo do Vizinho Mais Próximo (Nearest Neighbor) ou uma heurística
    baseada em A* para o Problema do Caixeiro Viajante (TSP) simplificado.
    
    O A* sozinho não resolve o TSP, mas pode ser usado para encontrar o
    caminho entre os pontos de um TSP resolvido por uma heurística.
    
    Aqui, vamos definir a estrutura para a otimização do TSP simplificado.
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.pathfinder = AStarPathfinder(graph)
        
    def _solve_tsp_nearest_neighbor(self, node_ids: List[int], start_node_id: int, cost_type: str = 'time') -> Tuple[List[int], float]:
        """
        Heurística do Vizinho Mais Próximo para o Problema do Caixeiro Viajante (TSP).
        
        Args:
            node_ids: Lista de IDs dos nós a serem visitados (excluindo o ponto de partida).
            start_node_id: ID do nó de partida (Restaurante/Depósito).
            cost_type: Tipo de custo a ser minimizado ('time' ou 'distance').
            
        Returns:
            Uma tupla contendo a lista de IDs dos nós na rota otimizada e o custo total.
        """
        
        # Pontos a serem visitados, incluindo o ponto de partida e o retorno
        all_points_to_visit = set(node_ids)
        current_node_id = start_node_id
        
        # A rota começa e termina no ponto de partida
        optimized_route = [start_node_id]
        total_cost = 0.0
        
        # Enquanto houver pontos a serem visitados
        while all_points_to_visit:
            best_next_node = -1
            min_cost = float('inf')
            best_path = []
            
            # Encontra o vizinho mais próximo entre os pontos restantes
            for next_node_id in all_points_to_visit:
                # Usa A* para encontrar o caminho e o custo entre o nó atual e o próximo
                path_result = self.pathfinder.find_shortest_path(current_node_id, next_node_id, cost_type)
                
                if path_result:
                    path, cost = path_result
                    if cost < min_cost:
                        min_cost = cost
                        best_next_node = next_node_id
                        best_path = path
            
            if best_next_node != -1:
                # Adiciona o caminho (excluindo o nó de partida do caminho)
                optimized_route.extend(best_path[1:])
                total_cost += min_cost
                current_node_id = best_next_node
                all_points_to_visit.remove(best_next_node)
            else:
                # Não foi possível encontrar um caminho para os nós restantes (grafo desconexo)
                break
                
        # Retorna ao ponto de partida (Restaurante/Depósito)
        path_result = self.pathfinder.find_shortest_path(current_node_id, start_node_id, cost_type)
        if path_result:
            path, cost = path_result
            optimized_route.extend(path[1:])
            total_cost += cost
            
        return optimized_route, total_cost

    def optimize_routes(self, orders: List[Order], couriers: List[Courier], n_clusters: int, cost_type: str = 'time') -> Dict[int, Tuple[List[int], float]]:
        """
        Orquestra o agrupamento e a otimização de rotas para múltiplos entregadores.
        
        Args:
            orders: Lista de todos os pedidos.
            couriers: Lista de entregadores disponíveis.
            n_clusters: Número de clusters (idealmente igual ao número de entregadores).
            cost_type: Tipo de custo a ser minimizado ('time' ou 'distance').
            
        Returns:
            Um dicionário onde a chave é o ID do cluster (e implicitamente do entregador)
            e o valor é uma tupla (rota otimizada, custo total).
        """
        
        # 1. Preparar dados para o K-Means
        # Coordenadas dos nós de destino dos pedidos
        order_nodes_coords = []
        order_node_ids = []
        for order in orders:
            node = self.graph.nodes.get(order.node_id)
            if node:
                order_nodes_coords.append([node.lat, node.lon])
                order_node_ids.append(order.node_id)
        
        if not order_nodes_coords:
            return {}
            
        data = np.array(order_nodes_coords)
        
        # 2. Executar K-Means
        # O número de clusters deve ser igual ao número de entregadores disponíveis
        # ou um número otimizado (e.g., Elbow Method), mas para simplificação, usamos n_clusters.
        kmeans = KMeansClustering(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit(data)
        
        # 3. Atribuir rótulos de cluster aos pedidos
        for i, order in enumerate(orders):
            order.cluster_id = int(cluster_labels[i])
            
        # 4. Otimizar a rota para cada cluster/entregador
        optimized_routes: Dict[int, Tuple[List[int], float]] = {}
        
        for cluster_id in range(n_clusters):
            # Encontra os nós de destino para este cluster
            cluster_node_ids = [order.node_id for order in orders if order.cluster_id == cluster_id]
            
            # Remove duplicatas, pois vários pedidos podem ir para o mesmo nó
            unique_cluster_node_ids = list(set(cluster_node_ids))
            
            if not unique_cluster_node_ids:
                continue
                
            # O ponto de partida é o Restaurante/Depósito (Nó 0)
            start_node_id = 0
            
            # Resolve o TSP simplificado para os nós neste cluster
            route, cost = self._solve_tsp_nearest_neighbor(unique_cluster_node_ids, start_node_id, cost_type)
            
            optimized_routes[cluster_id] = (route, cost)
            
        return optimized_routes

if __name__ == '__main__':
    from data_model import create_example_data
    
    graph, orders, couriers = create_example_data()
    
    # Teste do A*
    pathfinder = AStarPathfinder(graph)
    start_node = 0
    end_node = 3
    path_result = pathfinder.find_shortest_path(start_node, end_node, cost_type='distance')
    
    print(f"--- Teste do A* (Nó {start_node} para Nó {end_node}, Custo: Distância) ---")
    if path_result:
        path, cost = path_result
        print(f"Caminho: {path}")
        print(f"Distância Total: {cost:.4f} km")
    else:
        print("Caminho não encontrado.")
        
    # Teste do RouteOptimizer (K-Means + TSP Simplificado)
    print("\n--- Teste do RouteOptimizer (K-Means + TSP Simplificado) ---")
    
    # Usamos o número de entregadores como o número de clusters
    n_clusters = len(couriers) 
    optimizer = RouteOptimizer(graph)
    
    optimized_routes = optimizer.optimize_routes(orders, couriers, n_clusters, cost_type='time')
    
    print(f"Otimização de Rotas para {n_clusters} clusters/entregadores:")
    for cluster_id, (route, cost) in optimized_routes.items():
        courier_name = couriers[cluster_id].name if cluster_id < len(couriers) else f"Cluster {cluster_id}"
        print(f"\n{courier_name} (Cluster {cluster_id}):")
        
        # Mapeia IDs de nós para nomes para melhor visualização
        route_names = [graph.nodes[node_id].name for node_id in route]
        
        print(f"  Rota (IDs): {route}")
        print(f"  Rota (Nomes): {' -> '.join(route_names)}")
        print(f"  Tempo Total Estimado: {cost:.2f} minutos")
        
        # Verifica quais pedidos foram atribuídos a este cluster
        assigned_orders = [order.id for order in orders if order.cluster_id == cluster_id]
        print(f"  Pedidos Atribuídos (IDs): {assigned_orders}")
