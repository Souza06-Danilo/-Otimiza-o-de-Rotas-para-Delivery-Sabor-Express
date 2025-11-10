# evaluation.py

from data_model import create_example_data, haversine_distance
from algorithms import RouteOptimizer
from typing import List, Dict, Tuple, Any

def calculate_total_distance(graph, route: List[int]) -> float:
    """Calcula a distância total percorrida em uma rota."""
    total_distance = 0.0
    for i in range(len(route) - 1):
        node_id_a = route[i]
        node_id_b = route[i+1]
        
        node_a = graph.nodes[node_id_a]
        node_b = graph.nodes[node_id_b]
        
        # Usa a distância Haversine como uma aproximação da distância real da rota
        # (O A* já calcula o custo da aresta, mas para uma métrica de rota total,
        # é mais simples somar as distâncias entre os nós sequenciais do caminho final)
        # No entanto, o custo total já é retornado pelo RouteOptimizer, que usa o A*
        # para somar os custos das arestas. Vamos usar o custo retornado.
        
        # Para fins de validação, vamos garantir que o custo retornado pelo A* é o que usamos.
        # O RouteOptimizer retorna o custo total baseado no cost_type ('time' ou 'distance').
        pass # O custo total já está na tupla de retorno

def evaluate_solution(graph, orders, couriers, n_clusters: int) -> Dict[str, Any]:
    """Executa a otimização e coleta métricas de desempenho."""
    
    optimizer = RouteOptimizer(graph)
    
    # 1. Otimização baseada em TEMPO (principal métrica)
    optimized_routes_time = optimizer.optimize_routes(orders, couriers, n_clusters, cost_type='time')
    
    # 2. Otimização baseada em DISTÂNCIA (métrica de comparação)
    # Recria os dados para garantir que os clusters sejam reatribuídos
    graph_dist, orders_dist, couriers_dist = create_example_data()
    optimizer_dist = RouteOptimizer(graph_dist)
    optimized_routes_dist = optimizer_dist.optimize_routes(orders_dist, couriers_dist, n_clusters, cost_type='distance')
    
    # --- Coleta de Métricas ---
    
    metrics = {
        "n_clusters": n_clusters,
        "n_orders": len(orders),
        "n_couriers": len(couriers),
        "time_optimization": {
            "total_time_min": sum(cost for _, cost in optimized_routes_time.values()),
            "routes": optimized_routes_time,
            "orders_per_cluster": {cid: len([o for o in orders if o.cluster_id == cid]) for cid in range(n_clusters)}
        },
        "distance_optimization": {
            "total_distance_km": sum(cost for _, cost in optimized_routes_dist.values()),
            "routes": optimized_routes_dist,
            "orders_per_cluster": {cid: len([o for o in orders_dist if o.cluster_id == cid]) for cid in range(n_clusters)}
        }
    }
    
    return metrics

def print_evaluation_results(metrics: Dict[str, Any]):
    """Imprime os resultados da avaliação de forma estruturada."""
    
    print("--- Resultados da Avaliação da Solução de Otimização de Rotas ---")
    print(f"Total de Pedidos: {metrics['n_orders']}")
    print(f"Total de Entregadores/Clusters: {metrics['n_couriers']}")
    
    # Otimização por Tempo
    print("\n[1] Otimização Baseada em TEMPO (Métrica Principal)")
    print(f"Tempo Total Estimado (Soma das Rotas): {metrics['time_optimization']['total_time_min']:.2f} minutos")
    
    for cluster_id, (route, cost) in metrics['time_optimization']['routes'].items():
        courier_name = f"Entregador {cluster_id + 1}"
        orders_count = metrics['time_optimization']['orders_per_cluster'].get(cluster_id, 0)
        print(f"  - {courier_name} (Cluster {cluster_id}): {cost:.2f} min | {orders_count} pedidos")
        
    # Otimização por Distância
    print("\n[2] Otimização Baseada em DISTÂNCIA (Métrica de Comparação)")
    print(f"Distância Total Estimada (Soma das Rotas): {metrics['distance_optimization']['total_distance_km']:.2f} km")
    
    for cluster_id, (route, cost) in metrics['distance_optimization']['routes'].items():
        courier_name = f"Entregador {cluster_id + 1}"
        orders_count = metrics['distance_optimization']['orders_per_cluster'].get(cluster_id, 0)
        print(f"  - {courier_name} (Cluster {cluster_id}): {cost:.2f} km | {orders_count} pedidos")
        
    # Comparação de Agrupamento (K-Means)
    # Nota: O K-Means é estocástico, mas com random_state=42, os resultados devem ser consistentes.
    # No entanto, a otimização por tempo e distância pode levar a agrupamentos ligeiramente diferentes
    # se o custo de uma aresta for muito diferente em tempo vs. distância.
    
    # Para simplificar a comparação, vamos focar nas métricas totais.
    
    print("\n[3] Análise do Agrupamento (K-Means)")
    print("Distribuição de Pedidos por Cluster (Otimização por Tempo):")
    print(metrics['time_optimization']['orders_per_cluster'])
    
    # A rota detalhada e a análise de capacidade (VRP) seriam o próximo passo,
    # mas a solução atual já demonstra o conceito de agrupamento e roteamento.
    
    # Métrica de Eficiência: Tempo médio por pedido
    avg_time_per_order = metrics['time_optimization']['total_time_min'] / metrics['n_orders']
    print(f"\nTempo Médio de Rota por Pedido: {avg_time_per_order:.2f} minutos")


if __name__ == '__main__':
    graph, orders, couriers = create_example_data()
    n_clusters = len(couriers)
    
    metrics = evaluate_solution(graph, orders, couriers, n_clusters)
    print_evaluation_results(metrics)
