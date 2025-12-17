# ==========================================================================
# Sistema de Optimización de Rutas con Análisis de Datos
# ==========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# ------------------------------------------
st.set_page_config(
    page_title="TMS Logístico Profesional",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚚"
)

# Cargar configuración
@st.cache_resource
def load_config():
    return {
        "ciudad": "Managua, Nicaragua",
        "deposito_coords": (12.1094, -86.2372),
        "zonas_managua": [
            {"nombre": "Centro", "coords": (12.1094, -86.2372), "radio": 0.05},
            {"nombre": "Carretera Sur", "coords": (12.0889, -86.2442), "radio": 0.08},
            {"nombre": "Carretera Norte", "coords": (12.0900, -86.2000), "radio": 0.1},
            {"nombre": "Villa Fontana", "coords": (12.1250, -86.2700), "radio": 0.06},
            {"nombre": "Bello Horizonte", "coords": (12.1400, -86.2300), "radio": 0.07},
            {"nombre": "Metrocentro", "coords": (12.1297, -86.2605), "radio": 0.04},
            {"nombre": "Altamira", "coords": (12.1500, -86.2500), "radio": 0.08},
        ],
        "velocidad_promedio": 40,  # km/h
        "tiempo_servicio": 15,     # minutos
        "costo_por_km": 0.8,
        "costo_por_hora": 25.0
    }

config = load_config()

# ------------------------------------------
# CLASES PRINCIPALES DEL SISTEMA
# ------------------------------------------
class Cliente:
    """Clase para representar un cliente en el sistema"""
    
    def __init__(self, id: int, nombre: str, coordenadas: Tuple[float, float], 
                 demanda: int, ventana_tiempo: Tuple[int, int], 
                 tipo: str = "normal", prioridad: int = 1):
        self.id = id
        self.nombre = nombre
        self.coordenadas = coordenadas
        self.demanda = demanda
        self.ventana_tiempo = ventana_tiempo
        self.tipo = tipo
        self.prioridad = prioridad
        self.tiempo_servicio = config["tiempo_servicio"]
        
    def to_dict(self):
        return {
            "id": self.id,
            "nombre": self.nombre,
            "lat": self.coordenadas[0],
            "lon": self.coordenadas[1],
            "demanda": self.demanda,
            "ventana_inicio": self.ventana_tiempo[0],
            "ventana_fin": self.ventana_tiempo[1],
            "tipo": self.tipo,
            "prioridad": self.prioridad
        }

class Ruta:
    """Clase para representar una ruta de entrega"""
    
    def __init__(self, id_vehiculo: int, clientes: List[Cliente], 
                 matriz_distancias: np.ndarray):
        self.id_vehiculo = id_vehiculo
        self.clientes = clientes
        self.matriz_distancias = matriz_distancias
        self.distancia_total = self.calcular_distancia()
        self.tiempo_total = self.calcular_tiempo()
        self.demanda_total = sum(c.demanda for c in clientes)
        
    def calcular_distancia(self) -> float:
        """Calcula la distancia total de la ruta en metros"""
        distancia = 0
        for i in range(len(self.clientes) - 1):
            c1 = self.clientes[i]
            c2 = self.clientes[i + 1]
            distancia += self.matriz_distancias[c1.id][c2.id]
        return distancia
    
    def calcular_tiempo(self) -> float:
        """Calcula el tiempo total de la ruta en minutos"""
        tiempo_viaje = (self.distancia_total / 1000) / (config["velocidad_promedio"] / 60)
        tiempo_servicio = len(self.clientes) * config["tiempo_servicio"]
        return tiempo_viaje + tiempo_servicio
    
    def calcular_costo(self) -> float:
        """Calcula el costo total de la ruta"""
        costo_combustible = (self.distancia_total / 1000) * config["costo_por_km"]
        costo_tiempo = (self.tiempo_total / 60) * config["costo_por_hora"]
        return costo_combustible + costo_tiempo
    
    def to_dict(self):
        return {
            "vehiculo": self.id_vehiculo,
            "num_clientes": len(self.clientes),
            "distancia_km": round(self.distancia_total / 1000, 2),
            "tiempo_min": round(self.tiempo_total, 1),
            "demanda_total": self.demanda_total,
            "costo": round(self.calcular_costo(), 2),
            "utilizacion": round((self.demanda_total / st.session_state.get('capacidad_camion', 20)) * 100, 1)
        }

# ------------------------------------------
# GENERADOR DE DATOS INTELIGENTE
# ------------------------------------------
class GeneradorDatos:
    """Genera datos realistas para simulación"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
    def generar_clientes(self, num_clientes: int) -> Dict[int, Cliente]:
        """Genera clientes con distribución realista en Managua"""
        
        # Crear depósito (cliente 0)
        deposito = Cliente(
            id=0,
            nombre="CENTRO DE DISTRIBUCIÓN",
            coordenadas=config["deposito_coords"],
            demanda=0,
            ventana_tiempo=(0, 600),
            tipo="deposito",
            prioridad=0
        )
        
        clientes = {0: deposito}
        
        # Tipos de clientes con probabilidades realistas
        tipos = ["normal", "urgente", "programado", "empresa"]
        pesos_tipos = [0.6, 0.15, 0.15, 0.1]
        
        # Demandas con distribución realista
        demandas_posibles = [1, 2, 3, 4, 5]
        pesos_demandas = [0.1, 0.3, 0.4, 0.15, 0.05]
        
        for i in range(1, num_clientes + 1):
            # Seleccionar zona aleatoria
            zona = random.choice(config["zonas_managua"])
            
            # Generar coordenadas realistas dentro de la zona
            lat = zona["coords"][0] + random.uniform(-zona["radio"], zona["radio"])
            lon = zona["coords"][1] + random.uniform(-zona["radio"], zona["radio"])
            
            # Ventana de tiempo realista (8:00 AM - 6:00 PM)
            if random.random() < 0.3:  # 30% tienen ventanas tempranas
                inicio = random.randint(60, 180)  # 9:00 AM - 12:00 PM
            else:
                inicio = random.randint(180, 420)  # 12:00 PM - 4:00 PM
            
            duracion = random.choice([60, 90, 120])  # 1, 1.5 o 2 horas
            fin = min(600, inicio + duracion)  # Máximo 6:00 PM
            
            # Seleccionar tipo y demanda
            tipo = random.choices(tipos, pesos_tipos)[0]
            demanda = random.choices(demandas_posibles, pesos_demandas)[0]
            
            # Ajustar prioridad según tipo
            prioridad = 1
            if tipo == "urgente":
                prioridad = 3
            elif tipo == "empresa":
                prioridad = 2
            
            # Crear cliente
            cliente = Cliente(
                id=i,
                nombre=f"Cliente {i} - {zona['nombre']}",
                coordenadas=(lat, lon),
                demanda=demanda,
                ventana_tiempo=(inicio, fin),
                tipo=tipo,
                prioridad=prioridad
            )
            
            clientes[i] = cliente
        
        return clientes
    
    def generar_matriz_distancias(self, clientes: Dict[int, Cliente]) -> np.ndarray:
        """Genera matriz de distancias realista (simulada)"""
        n = len(clientes)
        matriz = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Distancia euclidiana aproximada en metros
                    lat1, lon1 = clientes[i].coordenadas
                    lat2, lon2 = clientes[j].coordenadas
                    
                    # Conversión aproximada a metros (1 grado ≈ 111 km)
                    distancia = np.sqrt(((lat1 - lat2) * 111000) ** 2 + 
                                       ((lon1 - lon2) * 111000 * np.cos(np.radians(lat1))) ** 2)
                    
                    # Añadir factor de red vial (30% más)
                    distancia *= 1.3
                    
                    # Redondear a metros
                    matriz[i][j] = int(distancia)
        
        return matriz

# ------------------------------------------
# ALGORITMO DE OPTIMIZACIÓN HÍBRIDO
# ------------------------------------------
class OptimizadorVRP:
    """Sistema de optimización de rutas híbrido"""
    
    def __init__(self):
        self.solucion = None
        self.rutas = []
        self.kpis = {}
    
    def optimizar_con_clustering(self, clientes: Dict[int, Cliente], 
                                matriz_distancias: np.ndarray, 
                                num_vehiculos: int, 
                                capacidad: int):
        """Optimización híbrida: clustering + VRP"""
        
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            st.error("❌ sklearn no está instalado. Ejecuta: pip install scikit-learn")
            return []
        
        # Extraer coordenadas de clientes (excluyendo depósito)
        if len(clientes) <= 1:
            return []
        
        coords = np.array([clientes[i].coordenadas for i in range(1, len(clientes))])
        
        # Asegurar que tenemos suficientes clientes para clustering
        if len(coords) < num_vehiculos:
            num_vehiculos = max(1, len(coords))
        
        # Aplicar K-means clustering
        try:
            kmeans = KMeans(n_clusters=num_vehiculos, random_state=42, n_init=10)
            etiquetas = kmeans.fit_predict(coords)
        except Exception as e:
            st.warning(f"⚠️ Error en clustering: {e}. Usando asignación simple.")
            # Asignación simple por proximidad
            return self.asignacion_simple(clientes, matriz_distancias, num_vehiculos, capacidad)
        
        # Crear rutas por cluster
        rutas = []
        
        for cluster_id in range(num_vehiculos):
            # Identificar clientes en este cluster
            ids_cluster = [0] + [i+1 for i, label in enumerate(etiquetas) if label == cluster_id]
            
            if len(ids_cluster) > 1:
                # Optimizar secuencia dentro del cluster (TSP)
                ruta_optima = self.resolver_tsp_cluster(ids_cluster, matriz_distancias)
                rutas.append(ruta_optima)
            elif len(ids_cluster) == 1 and ids_cluster[0] == 0:
                # Cluster vacío (solo depósito)
                continue
        
        # Ajustar balance de carga
        rutas_balanceadas = self.balancear_carga(rutas, clientes, capacidad, matriz_distancias)
        
        return rutas_balanceadas
    
    def asignacion_simple(self, clientes: Dict[int, Cliente], 
                         matriz_distancias: np.ndarray,
                         num_vehiculos: int,
                         capacidad: int):
        """Asignación simple cuando clustering falla"""
        rutas = []
        clientes_ids = list(clientes.keys())
        clientes_ids.remove(0)  # Remover depósito
        
        # Distribuir clientes equitativamente
        clientes_por_ruta = len(clientes_ids) // num_vehiculos
        resto = len(clientes_ids) % num_vehiculos
        
        inicio = 0
        for i in range(num_vehiculos):
            # Calcular cuántos clientes para esta ruta
            cantidad = clientes_por_ruta + (1 if i < resto else 0)
            if cantidad == 0:
                continue
                
            # Tomar clientes para esta ruta
            ids_ruta = [0] + clientes_ids[inicio:inicio+cantidad]
            
            # Ordenar por proximidad (algoritmo simple)
            ruta_ordenada = self.ordenar_por_proximidad(ids_ruta, matriz_distancias)
            rutas.append(ruta_ordenada)
            
            inicio += cantidad
        
        return rutas
    
    def ordenar_por_proximidad(self, ids_ruta: List[int], matriz_distancias: np.ndarray):
        """Ordena clientes por proximidad (algoritmo del vecino más cercano)"""
        if len(ids_ruta) <= 2:
            return ids_ruta
        
        # Iniciar con el depósito
        ruta_ordenada = [0]
        no_visitados = [c for c in ids_ruta if c != 0]
        
        while no_visitados:
            ultimo = ruta_ordenada[-1]
            
            # Encontrar el cliente más cercano al último
            distancias = [(matriz_distancias[ultimo][c], c) for c in no_visitados]
            distancias.sort()
            cliente_mas_cercano = distancias[0][1]
            
            # Añadir a la ruta
            ruta_ordenada.append(cliente_mas_cercano)
            no_visitados.remove(cliente_mas_cercano)
        
        # Volver al depósito
        ruta_ordenada.append(0)
        return ruta_ordenada
    
    def resolver_tsp_cluster(self, ids_cluster: List[int], 
                           matriz_distancias: np.ndarray) -> List[int]:
        """Resuelve TSP para un cluster usando algoritmo de inserción más barata"""
        
        if len(ids_cluster) <= 2:
            return ids_cluster
        
        # Algoritmo de inserción más barata
        ruta = [ids_cluster[0], ids_cluster[1]]
        
        for cliente_id in ids_cluster[2:]:
            mejor_posicion = 0
            mejor_incremento = float('inf')
            
            # Encontrar posición óptima
            for i in range(len(ruta) - 1):
                costo_actual = matriz_distancias[ruta[i]][ruta[i+1]]
                costo_nuevo = (matriz_distancias[ruta[i]][cliente_id] + 
                             matriz_distancias[cliente_id][ruta[i+1]])
                incremento = costo_nuevo - costo_actual
                
                if incremento < mejor_incremento:
                    mejor_incremento = incremento
                    mejor_posicion = i + 1
            
            # Insertar cliente en la posición óptima
            ruta.insert(mejor_posicion, cliente_id)
        
        return ruta
    
    def balancear_carga(self, rutas: List[List[int]], 
                       clientes: Dict[int, Cliente], 
                       capacidad: int,
                       matriz_distancias: np.ndarray) -> List[List[int]]:
        """Balancea la carga entre rutas para optimizar utilización"""
        
        if not rutas:
            return rutas
        
        # Calcular demanda por ruta
        demandas_rutas = []
        for ruta in rutas:
            demanda = sum(clientes[cliente_id].demanda for cliente_id in ruta if cliente_id != 0)
            demandas_rutas.append((demanda, ruta))
        
        # Ordenar rutas por demanda
        demandas_rutas.sort(key=lambda x: x[0])
        
        # Balancear moviendo clientes entre rutas
        rutas_balanceadas = [ruta for _, ruta in demandas_rutas]
        
        # Intentar mover clientes de rutas sobrecargadas a subcargadas
        for _ in range(10):  # 10 iteraciones de balanceo
            for i in range(len(rutas_balanceadas)):
                for j in range(len(rutas_balanceadas)):
                    if i != j:
                        demanda_i = sum(clientes[c].demanda for c in rutas_balanceadas[i] if c != 0)
                        demanda_j = sum(clientes[c].demanda for c in rutas_balanceadas[j] if c != 0)
                        
                        # Si hay desbalance significativo
                        if demanda_i > capacidad * 0.8 and demanda_j < capacidad * 0.6:
                            # Intentar mover un cliente de i a j
                            for cliente_id in rutas_balanceadas[i][1:-1]:  # Excluir depósito
                                if (demanda_i - clientes[cliente_id].demanda >= 0 and
                                    demanda_j + clientes[cliente_id].demanda <= capacidad):
                                    
                                    # Mover cliente
                                    rutas_balanceadas[i].remove(cliente_id)
                                    # Insertar en posición óptima en ruta j
                                    self.insertar_cliente_optimo(rutas_balanceadas[j], 
                                                                cliente_id, 
                                                                matriz_distancias)
                                    break
        
        return rutas_balanceadas
    
    def insertar_cliente_optimo(self, ruta: List[int], cliente_id: int, 
                              matriz_distancias: np.ndarray):
        """Inserta un cliente en la posición óptima de una ruta"""
        
        if not ruta:
            ruta.append(cliente_id)
            return
        
        mejor_posicion = 1
        mejor_costo = float('inf')
        
        for i in range(1, len(ruta)):
            costo_actual = matriz_distancias[ruta[i-1]][ruta[i]]
            costo_nuevo = (matriz_distancias[ruta[i-1]][cliente_id] + 
                         matriz_distancias[cliente_id][ruta[i]])
            incremento = costo_nuevo - costo_actual
            
            if incremento < mejor_costo:
                mejor_costo = incremento
                mejor_posicion = i
        
        ruta.insert(mejor_posicion, cliente_id)
    
    def calcular_kpis(self, rutas: List[List[int]], clientes: Dict[int, Cliente], 
                     matriz_distancias: np.ndarray) -> Dict:
        """Calcula KPIs clave del sistema"""
        
        # Obtener capacidad del camión del estado de sesión o usar valor por defecto
        capacidad_camion = st.session_state.get('capacidad_camion', 20)
        
        kpis = {
            "total_clientes": len(clientes) - 1,
            "vehiculos_utilizados": len([r for r in rutas if len(r) > 2]),
            "vehiculos_totales": st.session_state.get('num_camiones', 12),
            "distancia_total_km": 0,
            "tiempo_total_horas": 0,
            "demanda_total": 0,
            "costo_total": 0,
            "utilizacion_promedio": 0,
            "otr_promedio": 0.95  # On-Time Rate estimado
        }
        
        if not rutas:
            return kpis
        
        utilizaciones = []
        
        for ruta in rutas:
            if len(ruta) > 2:
                # Calcular métricas de la ruta
                distancia = 0
                demanda = 0
                
                for i in range(len(ruta) - 1):
                    distancia += matriz_distancias[ruta[i]][ruta[i+1]]
                    if ruta[i] != 0:
                        demanda += clientes[ruta[i]].demanda
                
                distancia_km = distancia / 1000
                tiempo_horas = distancia_km / config["velocidad_promedio"]
                tiempo_horas += (len(ruta) - 2) * config["tiempo_servicio"] / 60
                
                utilizacion = (demanda / capacidad_camion) * 100 if capacidad_camion > 0 else 0
                
                kpis["distancia_total_km"] += distancia_km
                kpis["tiempo_total_horas"] += tiempo_horas
                kpis["demanda_total"] += demanda
                kpis["costo_total"] += (distancia_km * config["costo_por_km"] + 
                                      tiempo_horas * config["costo_por_hora"])
                
                utilizaciones.append(utilizacion)
        
        if utilizaciones:
            kpis["utilizacion_promedio"] = np.mean(utilizaciones)
        
        return kpis

# ------------------------------------------
# SISTEMA DE VISUALIZACIÓN AVANZADO
# ------------------------------------------
class Visualizador:
    """Sistema de visualización de datos y mapas"""
    
    def __init__(self):
        self.colores_rutas = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
        ]
    
    def crear_mapa_calor(self, clientes: Dict[int, Cliente], 
                        rutas: List[List[int]]) -> go.Figure:
        """Crea mapa de calor de densidad de clientes"""
        
        # Extraer coordenadas
        lats = [c.coordenadas[0] for c in clientes.values() if c.id != 0]
        lons = [c.coordenadas[1] for c in clientes.values() if c.id != 0]
        demandas = [c.demanda for c in clientes.values() if c.id != 0]
        
        # Crear figura
        fig = go.Figure()
        
        # Mapa base
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=8,
                color=demandas,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Demanda")
            ),
            text=[c.nombre for c in clientes.values() if c.id != 0],
            name='Clientes'
        ))
        
        # Añadir rutas
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                ruta_lats = [clientes[node_id].coordenadas[0] for node_id in ruta]
                ruta_lons = [clientes[node_id].coordenadas[1] for node_id in ruta]
                
                fig.add_trace(go.Scattermapbox(
                    lat=ruta_lats,
                    lon=ruta_lons,
                    mode='lines',
                    line=dict(width=3, color=self.colores_rutas[idx % len(self.colores_rutas)]),
                    name=f'Vehículo {idx+1}',
                    showlegend=True
                ))
        
        # Añadir depósito
        fig.add_trace(go.Scattermapbox(
            lat=[clientes[0].coordenadas[0]],
            lon=[clientes[0].coordenadas[1]],
            mode='markers',
            marker=dict(size=12, color='black', symbol='circle'),
            name='Depósito',
            showlegend=True
        ))
        
        # Configurar layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=config["deposito_coords"][0], 
                          lon=config["deposito_coords"][1]),
                zoom=12
            ),
            height=600,
            title="Mapa de Calor y Rutas Optimizadas",
            legend=dict(x=0, y=1)
        )
        
        return fig
    
    def crear_grafico_progresion_rutas(self, rutas, clientes, matriz_distancias):
        """Gráfico heatmap de progresión temporal de rutas"""
        
        datos_heatmap = []
        max_horas = 8  # Jornada de 8 horas
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                tiempo_acumulado = 0
                segmentos_por_hora = [0] * max_horas
                
                for i in range(len(ruta) - 1):
                    distancia = matriz_distancias[ruta[i]][ruta[i+1]] / 1000
                    tiempo_viaje = (distancia / config["velocidad_promedio"]) * 60  # minutos
                    
                    if ruta[i] != 0:
                        tiempo_acumulado += config["tiempo_servicio"]
                    
                    # Agregar tiempo de viaje
                    tiempo_acumulado += tiempo_viaje
                    
                    # Distribuir en horas
                    hora_actual = min(int(tiempo_acumulado / 60), max_horas - 1)
                    segmentos_por_hora[hora_actual] += 1
                
                for hora, valor in enumerate(segmentos_por_hora):
                    datos_heatmap.append({
                        "Vehículo": f"V{idx+1}",
                        "Hora": hora,
                        "Segmentos": valor,
                        "Hora_Texto": f"{8+hora}:00-{8+hora}:59"
                    })
        
        if not datos_heatmap:
            fig = go.Figure()
            fig.update_layout(title="No hay datos para mostrar")
            return fig
        
        df = pd.DataFrame(datos_heatmap)
        pivot = df.pivot(index="Vehículo", columns="Hora_Texto", values="Segmentos")
        
        fig = px.imshow(
            pivot,
            labels=dict(x="Horario", y="Vehículo", color="Segmentos"),
            title="🌡️ Calendario de Actividad por Vehículo",
            color_continuous_scale="YlOrRd",
            aspect="auto",
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Horario del Día",
            yaxis_title="Vehículo",
            coloraxis_colorbar_title="Nº Segmentos"
        )
        
        return fig
    
    def crear_grafico_distribucion_tiempos(self, rutas, clientes, matriz_distancias):
        """Gráfico de violín que muestra distribución de tiempos por vehículo"""
        
        datos_tiempos = []
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                tiempo_total = 0
                tiempos_por_parada = []
                
                for i in range(len(ruta) - 1):
                    if ruta[i] != 0:  # No es depósito
                        # Tiempo de servicio en este cliente
                        tiempos_por_parada.append(config["tiempo_servicio"])
                    
                    # Tiempo de viaje al siguiente punto
                    distancia = matriz_distancias[ruta[i]][ruta[i+1]] / 1000
                    tiempo_viaje = (distancia / config["velocidad_promedio"]) * 60
                    tiempos_por_parada.append(tiempo_viaje)
                    tiempo_total += tiempo_viaje
                    
                    if ruta[i] != 0:
                        tiempo_total += config["tiempo_servicio"]
                
                # Datos agregados para este vehículo
                if tiempos_por_parada:
                    for tiempo in tiempos_por_parada:
                        datos_tiempos.append({
                            "Vehículo": f"V{idx+1}",
                            "Tipo": "Tiempo Parada" if tiempo == config["tiempo_servicio"] else "Tiempo Viaje",
                            "Minutos": tiempo,
                            "Tiempo Total (h)": round(tiempo_total/60, 1),
                            "Paradas": len(ruta) - 2
                        })
        
        if not datos_tiempos:
            fig = go.Figure()
            fig.update_layout(title="No hay datos para mostrar", height=400)
            return fig
        
        df = pd.DataFrame(datos_tiempos)
        
        # Gráfico de violín combinado con box plot
        fig = go.Figure()
        
        # Separar por tipo de tiempo
        for tipo in ["Tiempo Viaje", "Tiempo Parada"]:
            df_tipo = df[df["Tipo"] == tipo]
            
            if len(df_tipo) > 0:
                fig.add_trace(go.Violin(
                    x=df_tipo["Vehículo"],
                    y=df_tipo["Minutos"],
                    name=tipo,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.3,
                    pointpos=-1.5,
                    fillcolor='rgba(100, 149, 237, 0.3)' if tipo == "Tiempo Viaje" else 'rgba(255, 182, 193, 0.3)',
                    line_color='royalblue' if tipo == "Tiempo Viaje" else 'lightcoral',
                    legendgroup=tipo
                ))
        
        fig.update_layout(
            title="🎻 Distribución de Tiempos por Vehículo",
            xaxis_title="Vehículo",
            yaxis_title="Minutos",
            height=500,
            violingap=0.1,
            violingroupgap=0.1,
            showlegend=True,
            plot_bgcolor='white'
        )
        
        # Añadir líneas de referencia
        fig.add_hline(y=config["tiempo_servicio"], 
                    line_dash="dot", 
                    line_color="gray",
                    annotation_text=f"Tiempo Servicio: {config['tiempo_servicio']} min")
        
        fig.add_hline(y=15, 
                    line_dash="dot", 
                    line_color="orange",
                    annotation_text="Límite Viaje: 15 min")
        
        return fig
    
    def crear_grafico_radar_desempeno(self, rutas, clientes, matriz_distancias):
        """Gráfico de radar comparando múltiples métricas por vehículo"""
        
        datos_radar = []
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                distancia_total = 0
                tiempo_total = 0
                demanda_total = 0
                
                for i in range(len(ruta) - 1):
                    distancia = matriz_distancias[ruta[i]][ruta[i+1]]
                    distancia_total += distancia
                    tiempo_viaje = (distancia / 1000) / (config["velocidad_promedio"] / 60)
                    tiempo_total += tiempo_viaje
                    
                    if ruta[i] != 0:
                        demanda_total += clientes[ruta[i]].demanda
                        tiempo_total += config["tiempo_servicio"]
                
                # Calcular métricas normalizadas (0-100)
                distancia_km = distancia_total / 1000
                utilizacion = (demanda_total / st.session_state.capacidad_camion) * 100 if st.session_state.capacidad_camion > 0 else 0
                eficiencia = (len(ruta) - 2) / max(1, distancia_km) * 10  # Normalizada
                velocidad_prom = (distancia_km / max(0.1, tiempo_total/60)) * 10  # Normalizada
                densidad = (len(ruta) - 2) * 5  # Normalizada
                
                # Limitar a 100
                utilizacion = min(100, utilizacion)
                eficiencia = min(100, eficiencia)
                velocidad_prom = min(100, velocidad_prom)
                densidad = min(100, densidad)
                
                # Obtener color y convertirlo a rgba con opacidad
                color_hex = self.colores_rutas[idx % len(self.colores_rutas)]
                # Convertir hex a rgba
                color_rgb = self.hex_to_rgba(color_hex, 0.3)
                
                datos_radar.append({
                    "Vehículo": f"V{idx+1}",
                    "Utilización": utilizacion,
                    "Eficiencia": eficiencia,
                    "Velocidad Prom": velocidad_prom,
                    "Densidad Ruta": densidad,
                    "Clientes": len(ruta) - 2,
                    "Distancia (km)": round(distancia_km, 1),
                    "Demanda": demanda_total,
                    "Color": color_hex,
                    "Color_RGBA": color_rgb
                })
        
        if len(datos_radar) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Se necesitan al menos 2 vehículos para comparar",
                height=400
            )
            return fig
        
        df = pd.DataFrame(datos_radar)
        
        # Crear gráfico de radar
        fig = go.Figure()
        
        # Limitar a 5 vehículos para mayor claridad
        vehiculos_a_mostrar = min(5, len(df))
        df_top = df.head(vehiculos_a_mostrar)
        
        categorias = ['Utilización', 'Eficiencia', 'Velocidad Prom', 'Densidad Ruta']
        
        for _, row in df_top.iterrows():
            valores = [row[cat] for cat in categorias]
            # Cerrar el polígono
            valores.append(valores[0])
            
            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=categorias + [categorias[0]],
                name=f"{row['Vehículo']} ({row['Clientes']} clientes)",
                fill='toself',
                fillcolor=row['Color_RGBA'],
                line=dict(color=row['Color'], width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['0%', '25%', '50%', '75%', '100%']
                )
            ),
            title=f"📡 Perfil de Desempeño (Top {vehiculos_a_mostrar} vehículos)",
            height=500,
            showlegend=True,
            legend=dict(x=1.1, y=0.5)
        )
        
        return fig

    def hex_to_rgba(self, hex_color, alpha=1.0):
        """Convierte color hexadecimal a formato rgba"""
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        elif len(hex_color) == 3:
            r = int(hex_color[0]*2, 16)
            g = int(hex_color[1]*2, 16)
            b = int(hex_color[2]*2, 16)
        else:
            r, g, b = 100, 100, 100
        
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    def crear_grafico_sankey_recursos(self, rutas, clientes, matriz_distancias):
        """Gráfico de Sankey mostrando distribución de recursos"""
        
        # Calcular totales
        total_tiempo = 0
        total_viaje = 0
        total_servicio = 0
        total_vehiculos = 0
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                total_vehiculos += 1
                for i in range(len(ruta) - 1):
                    distancia = matriz_distancias[ruta[i]][ruta[i+1]] / 1000
                    tiempo_viaje = (distancia / config["velocidad_promedio"]) * 60
                    total_viaje += tiempo_viaje
                    total_tiempo += tiempo_viaje
                    
                    if ruta[i] != 0:
                        total_servicio += config["tiempo_servicio"]
                        total_tiempo += config["tiempo_servicio"]
        
        if total_tiempo == 0:
            fig = go.Figure()
            fig.update_layout(title="No hay datos para mostrar", height=400)
            return fig
        
        # Preparar datos para Sankey
        labels = [
            "Recursos Totales",
            "Tiempo Viaje",
            "Tiempo Servicio",
            "Vehículo Activo",
            "Vehículo Inactivo",
            "Eficiente",
            "Ineficiente"
        ]
        
        source = [0, 0, 1, 1, 2, 2, 3, 3]
        target = [1, 2, 3, 5, 3, 5, 4, 6]
        value = [
            total_viaje,
            total_servicio,
            total_viaje * 0.8,
            total_viaje * 0.2,
            total_servicio * 0.9,
            total_servicio * 0.1,
            total_vehiculos * 0.7,
            total_vehiculos * 0.3
        ]
        
        # Colores
        color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8F71', '#3A7D44', '#9D4EDD']
        node_colors = color_palette[:len(labels)]
        link_colors = ['rgba(46, 134, 171, 0.4)', 'rgba(162, 59, 114, 0.4)', 
                    'rgba(241, 143, 1, 0.4)', 'rgba(199, 62, 29, 0.4)',
                    'rgba(107, 143, 113, 0.4)', 'rgba(58, 125, 68, 0.4)',
                    'rgba(157, 78, 221, 0.4)', 'rgba(46, 134, 171, 0.4)']
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
                hovertemplate='De %{source.label}<br>' +
                            'A %{target.label}<br>' +
                            'Valor: %{value:.0f} min<br>' +
                            '<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title_text="Flujo de Recursos y Tiempos",
            font_size=12,
            height=500
        )
        
        return fig
    
    def crear_grafico_burbujas_desempeno(self, rutas, clientes, matriz_distancias):
        """Gráfico de burbujas: Eficiencia vs Utilización por vehículo"""
        
        datos_burbujas = []
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                distancia_total = 0
                tiempo_total = 0
                demanda_total = 0
                
                for i in range(len(ruta) - 1):
                    distancia = matriz_distancias[ruta[i]][ruta[i+1]]
                    distancia_total += distancia
                    tiempo_viaje = (distancia / 1000) / (config["velocidad_promedio"] / 60)
                    tiempo_total += tiempo_viaje
                    
                    if ruta[i] != 0:
                        demanda_total += clientes[ruta[i]].demanda
                        tiempo_total += config["tiempo_servicio"]
                
                # Métricas de desempeño
                distancia_km = distancia_total / 1000
                utilizacion = (demanda_total / st.session_state.capacidad_camion) * 100 if st.session_state.capacidad_camion > 0 else 0
                
                # Eficiencia = clientes por km
                eficiencia = (len(ruta) - 2) / distancia_km if distancia_km > 0 else 0
                
                # Tiempo por cliente
                tiempo_por_cliente = tiempo_total / (len(ruta) - 2) if (len(ruta) - 2) > 0 else 0
                
                datos_burbujas.append({
                    "Vehículo": f"Vehículo {idx+1}",
                    "Utilización (%)": round(utilizacion, 1),
                    "Eficiencia (clientes/km)": round(eficiencia, 2),
                    "Tiempo por Cliente (min)": round(tiempo_por_cliente, 1),
                    "Clientes": len(ruta) - 2,
                    "Distancia (km)": round(distancia_km, 1),
                    "Color": self.colores_rutas[idx % len(self.colores_rutas)]
                })
        
        if not datos_burbujas:
            fig = go.Figure()
            fig.update_layout(title="No hay datos para mostrar")
            return fig
        
        df = pd.DataFrame(datos_burbujas)
        
        fig = px.scatter(
            df,
            x="Utilización (%)",
            y="Eficiencia (clientes/km)",
            size="Clientes",
            color="Vehículo",
            hover_name="Vehículo",
            hover_data=["Distancia (km)", "Tiempo por Cliente (min)"],
            title="📊 Matriz de Desempeño por Vehículo",
            size_max=50,
            height=500
        )
        
        # Añadir líneas de referencia
        fig.add_hline(y=df["Eficiencia (clientes/km)"].mean(), 
                    line_dash="dash", line_color="gray", 
                    annotation_text="Media Eficiencia", 
                    annotation_position="bottom right")
        
        fig.add_vline(x=70, line_dash="dash", line_color="red",
                    annotation_text="70% Utilización Óptima", 
                    annotation_position="top right")
        
        fig.add_vline(x=90, line_dash="dash", line_color="orange",
                    annotation_text="90% Límite Alto", 
                    annotation_position="top right")
        
        fig.update_layout(
            xaxis_title="Utilización (%)",
            yaxis_title="Eficiencia (clientes por km)",
            plot_bgcolor='rgba(240,240,240,0.5)',
            legend_title="Vehículos"
        )
        
        # Añadir zonas de desempeño
        fig.add_shape(type="rect",
                    x0=70, x1=90, y0=0, y1=df["Eficiencia (clientes/km)"].max() * 1.1,
                    fillcolor="green", opacity=0.1, line_width=0,
                    layer="below")
        
        fig.add_shape(type="rect",
                    x0=0, x1=70, y0=0, y1=df["Eficiencia (clientes/km)"].max() * 1.1,
                    fillcolor="yellow", opacity=0.1, line_width=0,
                    layer="below")
        
        fig.add_shape(type="rect",
                    x0=90, x1=100, y0=0, y1=df["Eficiencia (clientes/km)"].max() * 1.1,
                    fillcolor="red", opacity=0.1, line_width=0,
                    layer="below")
        
        return fig
    
    def crear_dashboard_kpis(self, kpis: Dict) -> go.Figure:
        """Crea dashboard visual con KPIs"""
        
        # Definir colores y temas
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        iconos = ['🚚', '🛣️', '⏱️', '💰', '⚡', '🎯']
        
        # Definir métricas con formatos específicos
        metricas = [
            {
                "nombre": "Vehículos",
                "valor": f"{kpis['vehiculos_utilizados']}/{kpis['vehiculos_totales']}",
                "subtitulo": f"{kpis['vehiculos_utilizados']/kpis['vehiculos_totales']*100:.1f}% utilización" if kpis['vehiculos_totales'] > 0 else "0% utilización",
                "color": '#667eea',
                "icono": '🚚',
                "formato": "fraccion"
            },
            {
                "nombre": "Distancia",
                "valor": f"{kpis['distancia_total_km']:.1f}",
                "subtitulo": "Kilómetros totales",
                "color": '#764ba2',
                "icono": '🛣️',
                "formato": "distancia"
            },
            {
                "nombre": "Tiempo",
                "valor": f"{kpis['tiempo_total_horas']:.1f}",
                "subtitulo": "Horas totales",
                "color": '#f093fb',
                "icono": '⏱️',
                "formato": "tiempo"
            },
            {
                "nombre": "Costo",
                "valor": f"{kpis['costo_total']:.2f}",
                "subtitulo": f"${kpis['costo_total']/max(1, kpis['total_clientes']):.2f}/cliente",
                "color": '#f5576c',
                "icono": '💰',
                "formato": "dinero"
            },
            {
                "nombre": "Utilización",
                "valor": f"{kpis['utilizacion_promedio']:.1f}",
                "subtitulo": "Porcentaje promedio",
                "color": '#4facfe',
                "icono": '⚡',
                "formato": "porcentaje"
            },
            {
                "nombre": "OTR",
                "valor": f"{kpis['otr_promedio']*100:.1f}",
                "subtitulo": "On-Time Rate",
                "color": '#43e97b',
                "icono": '🎯',
                "formato": "porcentaje"
            }
        ]
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Añadir cada indicador
        for i, metrica in enumerate(metricas):
            fila = i // 3 + 1
            columna = i % 3 + 1
            
            # Determinar formato del valor
            valor_num = 0
            formato_num = {}
            
            if metrica['formato'] == 'fraccion':
                partes = metrica['valor'].split('/')
                valor_num = float(partes[0]) / float(partes[1]) * 100 if len(partes) > 1 and float(partes[1]) > 0 else 0
                formato_num = {
                    'suffix': '%',
                    'valueformat': '.1f',
                    'font': {'size': 36, 'color': metrica['color'], 'family': 'Arial Black'}
                }
            elif metrica['formato'] == 'distancia':
                valor_num = float(metrica['valor'])
                formato_num = {
                    'suffix': ' km',
                    'valueformat': '.1f',
                    'font': {'size': 36, 'color': metrica['color'], 'family': 'Arial Black'}
                }
            elif metrica['formato'] == 'tiempo':
                valor_num = float(metrica['valor'])
                formato_num = {
                    'suffix': ' h',
                    'valueformat': '.1f',
                    'font': {'size': 36, 'color': metrica['color'], 'family': 'Arial Black'}
                }
            elif metrica['formato'] == 'dinero':
                valor_num = float(metrica['valor'])
                formato_num = {
                    'prefix': '$',
                    'valueformat': '.2f',
                    'font': {'size': 36, 'color': metrica['color'], 'family': 'Arial Black'}
                }
            elif metrica['formato'] == 'porcentaje':
                valor_num = float(metrica['valor'])
                formato_num = {
                    'suffix': '%',
                    'valueformat': '.1f',
                    'font': {'size': 36, 'color': metrica['color'], 'family': 'Arial Black'}
                }
            
            # Añadir indicador al subplot
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=valor_num,
                    number=formato_num,
                    title={
                        'text': f"<b>{metrica['icono']} {metrica['nombre']}</b><br>"
                               f"<span style='font-size:12px;color:#666'>{metrica['subtitulo']}</span>",
                        'font': {'size': 14, 'family': 'Arial'},
                        'align': 'center'
                    },
                    domain={'row': fila, 'column': columna}
                ),
                row=fila, col=columna
            )
            
            # Añadir valor textual adicional para fracción
            if metrica['formato'] == 'fraccion':
                fig.add_annotation(
                    x=columna/3 - 0.16,
                    y=1.1 - (fila-1)*0.5,
                    text=f"<span style='font-size:14px;color:#888'>{metrica['valor']}</span>",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    font=dict(size=12, color='#666')
                )
        
        # Actualizar layout
        fig.update_layout(
            height=500,
            title={
                'text': "📊 DASHBOARD DE KPIs LOGÍSTICOS",
                'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial Black'},
                'x': 0.5,
                'y': 0.97,
                'xanchor': 'center'
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=120, b=20, l=20, r=20),
            showlegend=False
        )
        
        # Añadir líneas divisorias
        fig.add_shape(
            type="line",
            x0=0.33, x1=0.33, y0=0, y1=1,
            line=dict(color="#e0e0e0", width=1, dash="dot"),
            xref="paper", yref="paper"
        )
        
        fig.add_shape(
            type="line",
            x0=0.67, x1=0.67, y0=0, y1=1,
            line=dict(color="#e0e0e0", width=1, dash="dot"),
            xref="paper", yref="paper"
        )
        
        fig.add_shape(
            type="line",
            x0=0, x1=1, y0=0.5, y1=0.5,
            line=dict(color="#e0e0e0", width=1, dash="dot"),
            xref="paper", yref="paper"
        )
        
        return fig
    
    def crear_grafico_utilizacion(self, rutas: List[List[int]], 
                                clientes: Dict[int, Cliente]) -> go.Figure:
        """Crea gráfico de utilización por vehículo"""
        
        datos = []
        capacidad_camion = st.session_state.get('capacidad_camion', 20)
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                demanda = sum(clientes[node_id].demanda for node_id in ruta if node_id != 0)
                utilizacion = (demanda / capacidad_camion) * 100 if capacidad_camion > 0 else 0
                datos.append({
                    "Vehículo": idx + 1,
                    "Utilización (%)": round(utilizacion, 1),
                    "Clientes": len(ruta) - 2,
                    "Color": self.colores_rutas[idx % len(self.colores_rutas)]
                })
        
        if not datos:
            fig = go.Figure()
            fig.update_layout(
                title="No hay rutas para mostrar",
                xaxis_title="Vehículo",
                yaxis_title="Utilización (%)",
                height=400
            )
            return fig
        
        df = pd.DataFrame(datos)
        
        fig = px.bar(
            df,
            x="Vehículo",
            y="Utilización (%)",
            color="Utilización (%)",
            text="Clientes",
            title="Utilización por Vehículo",
            color_continuous_scale="Viridis",
            height=400
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_title="Vehículo", yaxis_title="Utilización (%)")
        
        return fig
    
    def crear_grafico_distribucion_clientes(self, clientes: Dict[int, Cliente]) -> go.Figure:
        """Crea gráfico de distribución de tipos de cliente"""
        
        tipos = {}
        for cliente in clientes.values():
            if cliente.id != 0:
                tipos[cliente.tipo] = tipos.get(cliente.tipo, 0) + 1
        
        if not tipos:
            fig = go.Figure()
            fig.update_layout(
                title="No hay clientes para mostrar",
                height=400
            )
            return fig
        
        fig = px.pie(
            values=list(tipos.values()),
            names=list(tipos.keys()),
            title="Distribución de Tipos de Cliente",
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=400
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hole=0.3
        )
        
        return fig

# ------------------------------------------
# SISTEMA DE MAPAS INTERACTIVOS
# ------------------------------------------
class MapaInteractivo:
    """Sistema de mapas interactivos con folium"""
    
    def __init__(self):
        self.colores_rutas = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def crear_mapa_folium(self, clientes: Dict[int, Cliente], 
                         rutas: List[List[int]], 
                         ruta_seleccionada: int = None):
        """Crea un mapa interactivo con folium"""
        
        try:
            import folium
            from streamlit_folium import st_folium
        except ImportError:
            st.error("❌ Falta instalar streamlit-folium y folium. Ejecuta: pip install streamlit-folium folium")
            return None
        
        # Crear mapa centrado en el depósito
        mapa = folium.Map(
            location=config["deposito_coords"],
            zoom_start=12,
            control_scale=True,
            tiles='OpenStreetMap'
        )
        
        # Añadir capa de satélite
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satélite',
            overlay=False,
            control=True
        ).add_to(mapa)
        
        # Añadir capa de calles
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='Calles',
            overlay=False,
            control=True
        ).add_to(mapa)
        
        # Añadir capa de relieve
        folium.TileLayer(
            tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
            attr='Stamen Terrain',
            name='Relieve',
            overlay=False,
            control=True
        ).add_to(mapa)
        
        # Añadir depósito
        folium.Marker(
            location=config["deposito_coords"],
            popup=f"<b>🏭 CENTRO DE DISTRIBUCIÓN</b><br>"
                  f"ID: 0<br>"
                  f"Tipo: Depósito",
            tooltip="Centro de Distribución",
            icon=folium.Icon(color='black', icon='warehouse', prefix='fa')
        ).add_to(mapa)
        
        # Agrupar clientes por ruta para mejor visualización
        grupos_rutas = {}
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                grupos_rutas[idx] = ruta
        
        # Si hay una ruta seleccionada, mostrar solo esa ruta
        if ruta_seleccionada is not None and ruta_seleccionada in grupos_rutas:
            rutas_a_mostrar = {ruta_seleccionada: grupos_rutas[ruta_seleccionada]}
        else:
            rutas_a_mostrar = grupos_rutas
        
        # Añadir clientes y rutas
        for idx, ruta in rutas_a_mostrar.items():
            color_ruta = self.colores_rutas[idx % len(self.colores_rutas)]
            
            # Crear grupo de capas para esta ruta
            grupo_ruta = folium.FeatureGroup(name=f'Ruta {idx+1}', show=True)
            
            # Añadir línea de la ruta
            puntos_ruta = []
            for cliente_id in ruta:
                cliente = clientes[cliente_id]
                puntos_ruta.append(cliente.coordenadas)
            
            folium.PolyLine(
                puntos_ruta,
                color=color_ruta,
                weight=4,
                opacity=0.8,
                popup=f'<b>Ruta {idx+1}</b><br>'
                      f'Clientes: {len(ruta)-2}<br>'
                      f'Color: {color_ruta}',
                tooltip=f'Ruta {idx+1}'
            ).add_to(grupo_ruta)
            
            # Añadir marcadores de clientes en esta ruta
            for cliente_id in ruta:
                if cliente_id != 0:  # No incluir depósito (ya está añadido)
                    cliente = clientes[cliente_id]
                    
                    # Determinar icono según tipo de cliente
                    icono_color = 'green'
                    icono_tipo = 'user'
                    
                    if cliente.tipo == 'urgente':
                        icono_color = 'red'
                        icono_tipo = 'exclamation-triangle'
                    elif cliente.tipo == 'empresa':
                        icono_color = 'blue'
                        icono_tipo = 'building'
                    elif cliente.tipo == 'programado':
                        icono_color = 'orange'
                        icono_tipo = 'calendar-check'
                    
                    # Crear popup informativo
                    popup_html = f"""
                    <div style="font-family: Arial; width: 250px;">
                        <h4 style="margin:0; color: {icono_color};">📦 {cliente.nombre}</h4>
                        <hr style="margin: 5px 0;">
                        <p style="margin: 2px 0;"><strong>ID:</strong> {cliente.id}</p>
                        <p style="margin: 2px 0;"><strong>Demanda:</strong> {cliente.demanda} unidades</p>
                        <p style="margin: 2px 0;"><strong>Tipo:</strong> {cliente.tipo}</p>
                        <p style="margin: 2px 0;"><strong>Prioridad:</strong> {'★' * cliente.prioridad}</p>
                        <p style="margin: 2px 0;"><strong>Ventana:</strong> {cliente.ventana_tiempo[0]//60}:{str(cliente.ventana_tiempo[0]%60).zfill(2)} - {cliente.ventana_tiempo[1]//60}:{str(cliente.ventana_tiempo[1]%60).zfill(2)}</p>
                        <p style="margin: 2px 0;"><strong>Ruta:</strong> {idx+1}</p>
                    </div>
                    """
                    
                    folium.Marker(
                        location=cliente.coordenadas,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"Cliente {cliente.id}: {cliente.nombre}",
                        icon=folium.Icon(color=icono_color, icon=icono_tipo, prefix='fa')
                    ).add_to(grupo_ruta)
            
            # Añadir grupo al mapa
            grupo_ruta.add_to(mapa)
        
        # Añadir control de capas
        folium.LayerControl().add_to(mapa)
        
        # Añadir minimapa
        from folium.plugins import MiniMap
        minimap = MiniMap(toggle_display=True)
        mapa.add_child(minimap)
        
        # Añadir control de pantalla completa
        from folium.plugins import Fullscreen
        Fullscreen().add_to(mapa)
        
        # Añadir medición de distancias
        from folium.plugins import MeasureControl
        measure = MeasureControl()
        measure.add_to(mapa)
        
        # Añadir búsqueda de ubicaciones
        from folium.plugins import Geocoder
        Geocoder().add_to(mapa)
        
        return mapa
    
    def crear_mapa_calor_densidad(self, clientes: Dict[int, Cliente]):
        """Crea mapa de calor de densidad de clientes"""
        
        try:
            import folium
            from folium.plugins import HeatMap
        except ImportError:
            st.error("❌ Falta instalar folium. Ejecuta: pip install folium")
            return None
        
        # Preparar datos para el mapa de calor
        puntos_calor = []
        for cliente in clientes.values():
            if cliente.id != 0:
                # Peso según demanda y prioridad
                peso = cliente.demanda * (1 + cliente.prioridad * 0.5)
                puntos_calor.append([cliente.coordenadas[0], cliente.coordenadas[1], peso])
        
        # Crear mapa
        mapa = folium.Map(
            location=config["deposito_coords"],
            zoom_start=12,
            control_scale=True
        )
        
        # Añadir mapa de calor
        HeatMap(
            puntos_calor,
            min_opacity=0.3,
            max_zoom=15,
            radius=20,
            blur=15,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
        ).add_to(mapa)
        
        # Añadir depósito
        folium.Marker(
            location=config["deposito_coords"],
            popup="Centro de Distribución",
            icon=folium.Icon(color='black', icon='warehouse', prefix='fa')
        ).add_to(mapa)
        
        # Añadir clientes como marcadores
        for cliente in clientes.values():
            if cliente.id != 0:
                folium.CircleMarker(
                    location=cliente.coordenadas,
                    radius=5,
                    popup=f"{cliente.nombre}<br>Demanda: {cliente.demanda}",
                    color='blue',
                    fill=True,
                    fill_color='blue'
                ).add_to(mapa)
        
        return mapa
    
    def crear_mapa_clusters(self, clientes: Dict[int, Cliente], rutas: List[List[int]]):
        """Crea mapa con clusters de clientes"""
        
        try:
            import folium
            from folium.plugins import MarkerCluster
        except ImportError:
            st.error("❌ Falta instalar folium. Ejecuta: pip install folium")
            return None
        
        mapa = folium.Map(
            location=config["deposito_coords"],
            zoom_start=12,
            control_scale=True
        )
        
        # Crear cluster de marcadores
        marker_cluster = MarkerCluster().add_to(mapa)
        
        # Añadir depósito
        folium.Marker(
            location=config["deposito_coords"],
            popup="Centro de Distribución",
            icon=folium.Icon(color='black', icon='warehouse', prefix='fa')
        ).add_to(mapa)
        
        # Añadir clientes al cluster
        for cliente in clientes.values():
            if cliente.id != 0:
                folium.Marker(
                    location=cliente.coordenadas,
                    popup=f"<b>{cliente.nombre}</b><br>"
                          f"Demanda: {cliente.demanda}<br>"
                          f"Tipo: {cliente.tipo}",
                    icon=folium.Icon(color='blue', icon='user', prefix='fa')
                ).add_to(marker_cluster)
        
        # Añadir rutas
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                puntos_ruta = [clientes[cliente_id].coordenadas for cliente_id in ruta]
                folium.PolyLine(
                    puntos_ruta,
                    color=self.colores_rutas[idx % len(self.colores_rutas)],
                    weight=3,
                    opacity=0.7,
                    popup=f'Ruta {idx+1}'
                ).add_to(mapa)
        
        # Añadir control de capas
        folium.LayerControl().add_to(mapa)
        
        return mapa

# ------------------------------------------
# SISTEMA DE ALERTAS INTELIGENTES
# ------------------------------------------
class SistemaAlertas:
    """Sistema de alertas y notificaciones inteligentes"""
    
    def __init__(self):
        self.alertas = []
    
    def verificar_rutas(self, rutas: List[List[int]], 
                       clientes: Dict[int, Cliente], 
                       capacidad: int):
        """Verifica problemas potenciales en las rutas"""
        
        self.alertas = []
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                # Calcular demanda total
                demanda_total = sum(clientes[node_id].demanda for node_id in ruta if node_id != 0)
                
                # Alerta 1: Sobreutilización
                if demanda_total > capacidad:
                    self.alertas.append({
                        "tipo": "🚨",
                        "mensaje": f"Vehículo {idx+1} excede capacidad: {demanda_total}/{capacidad}",
                        "prioridad": "alta",
                        "accion": "Redistribuir carga"
                    })
                
                # Alerta 2: Subutilización
                elif demanda_total < capacidad * 0.4:
                    self.alertas.append({
                        "tipo": "⚠️",
                        "mensaje": f"Vehículo {idx+1} subutilizado: {demanda_total}/{capacidad}",
                        "prioridad": "media",
                        "accion": "Combinar rutas"
                    })
                
                # Alerta 3: Rutas muy largas
                if len(ruta) > 10:  # Más de 10 paradas
                    self.alertas.append({
                        "tipo": "⏱️",
                        "mensaje": f"Vehículo {idx+1} tiene {len(ruta)-2} paradas (muy extensa)",
                        "prioridad": "baja",
                        "accion": "Dividir ruta"
                    })
        
        # Alerta 4: Vehículos no utilizados
        rutas_activas = len([r for r in rutas if len(r) > 2])
        total_rutas = len(rutas)
        if rutas_activas < total_rutas:
            self.alertas.append({
                "tipo": "🚛",
                "mensaje": f"{total_rutas - rutas_activas} vehículos no utilizados",
                "prioridad": "media",
                "accion": "Reasignar o liberar"
            })
        
        return self.alertas
    
    def mostrar_alertas(self):
        """Muestra las alertas en la interfaz"""
        
        if not self.alertas:
            st.sidebar.success("✅ No hay alertas críticas")
            return
        
        st.sidebar.markdown("### 🔔 Alertas del Sistema")
        
        # Ordenar por prioridad
        prioridad_orden = {"alta": 3, "media": 2, "baja": 1}
        alertas_ordenadas = sorted(self.alertas, 
                                  key=lambda x: prioridad_orden.get(x["prioridad"], 0), 
                                  reverse=True)
        
        for alerta in alertas_ordenadas:
            color = "red" if alerta["prioridad"] == "alta" else "orange" if alerta["prioridad"] == "media" else "blue"
            
            st.sidebar.markdown(f"""
            <div style="
                border-left: 4px solid {color};
                padding: 10px;
                margin: 5px 0;
                background-color: #f8f9fa;
                border-radius: 5px;
            ">
                <strong>{alerta['tipo']} {alerta['mensaje']}</strong><br>
                <small>Acción: {alerta['accion']}</small>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# SIMULADOR WHAT-IF
# ------------------------------------------
class SimuladorWhatIf:
    """Simulador de escenarios para toma de decisiones"""
    
    def __init__(self):
        self.escenarios = {
            "base": "Escenario Actual",
            "aumento_demanda": "Aumento Demanda 20%",
            "reduccion_flota": "Reducción Flota 30%",
            "horario_extendido": "Horario Extendido",
            "cliente_prioritario": "Cliente Prioritario",
            "trafico_intenso": "Tráfico Intenso"
        }
    
    def simular_escenario(self, escenario: str, kpis_base: Dict, 
                         clientes: Dict[int, Cliente]) -> Dict:
        """Simula diferentes escenarios"""
        
        kpis_simulados = kpis_base.copy()
        
        if escenario == "aumento_demanda":
            # Aumentar demanda en 20%
            kpis_simulados["demanda_total"] *= 1.2
            kpis_simulados["costo_total"] *= 1.15
            kpis_simulados["utilizacion_promedio"] = min(100, kpis_simulados["utilizacion_promedio"] * 1.2)
            
        elif escenario == "reduccion_flota":
            # Reducir flota en 30%
            nuevos_vehiculos = max(1, int(kpis_base["vehiculos_totales"] * 0.7))
            kpis_simulados["vehiculos_totales"] = nuevos_vehiculos
            kpis_simulados["utilizacion_promedio"] = min(100, kpis_base["utilizacion_promedio"] * 1.4)
            kpis_simulados["costo_total"] *= 0.85
            
        elif escenario == "horario_extendido":
            # Extender horario (más ventanas de tiempo)
            kpis_simulados["tiempo_total_horas"] *= 1.3
            kpis_simulados["costo_total"] *= 1.2
            kpis_simulados["otr_promedio"] = min(0.99, kpis_base["otr_promedio"] * 1.1)
            
        elif escenario == "trafico_intenso":
            # Reducir velocidad por tráfico
            velocidad_actual = config["velocidad_promedio"]
            nueva_velocidad = velocidad_actual * 0.7
            factor_tiempo = velocidad_actual / nueva_velocidad
            
            kpis_simulados["tiempo_total_horas"] *= factor_tiempo
            kpis_simulados["costo_total"] *= 1.25
            kpis_simulados["otr_promedio"] = max(0.7, kpis_base["otr_promedio"] * 0.8)
        
        return kpis_simulados
    
    def mostrar_comparativa(self, kpis_base: Dict, kpis_simulados: Dict, 
                          nombre_escenario: str):
        """Muestra comparativa de escenarios"""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Costo Base", f"${kpis_base['costo_total']:.2f}")
        
        with col2:
            delta_costo = ((kpis_simulados['costo_total'] - kpis_base['costo_total']) / 
                          kpis_base['costo_total'] * 100) if kpis_base['costo_total'] > 0 else 0
            st.metric(f"Costo {nombre_escenario}", 
                     f"${kpis_simulados['costo_total']:.2f}", 
                     f"{delta_costo:+.1f}%")
        
        with col3:
            delta_tiempo = ((kpis_simulados['tiempo_total_horas'] - kpis_base['tiempo_total_horas']) / 
                           kpis_base['tiempo_total_horas'] * 100) if kpis_base['tiempo_total_horas'] > 0 else 0
            st.metric("Impacto Tiempo", 
                     f"{kpis_simulados['tiempo_total_horas']:.1f} h", 
                     f"{delta_tiempo:+.1f}%")

# ------------------------------------------
# SISTEMA DE EXPORTACIÓN
# ------------------------------------------
class SistemaExportacion:
    """Sistema de exportación de reportes y datos"""
    
    @staticmethod
    def exportar_csv(clientes: Dict[int, Cliente], rutas: List[List[int]], 
                    kpis: Dict) -> Dict:
        """Exporta datos a formato CSV"""
        
        # Datos de clientes
        datos_clientes = []
        for cliente in clientes.values():
            datos_clientes.append({
                "ID": cliente.id,
                "Nombre": cliente.nombre,
                "Latitud": cliente.coordenadas[0],
                "Longitud": cliente.coordenadas[1],
                "Demanda": cliente.demanda,
                "Ventana_Inicio": cliente.ventana_tiempo[0],
                "Ventana_Fin": cliente.ventana_tiempo[1],
                "Tipo": cliente.tipo,
                "Prioridad": cliente.prioridad
            })
        
        # Datos de rutas
        datos_rutas = []
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                for pos, cliente_id in enumerate(ruta):
                    if cliente_id != 0:
                        datos_rutas.append({
                            "Vehículo": idx + 1,
                            "Orden_Parada": pos,
                            "Cliente_ID": cliente_id,
                            "Cliente_Nombre": clientes[cliente_id].nombre
                        })
        
        # KPIs
        datos_kpis = [kpis]
        
        return {
            "clientes": pd.DataFrame(datos_clientes),
            "rutas": pd.DataFrame(datos_rutas),
            "kpis": pd.DataFrame(datos_kpis)
        }
    
    @staticmethod
    def generar_reporte_html(clientes: Dict[int, Cliente], rutas: List[List[int]], 
                           kpis: Dict, matriz_distancias: np.ndarray) -> str:
        """Genera reporte HTML ejecutivo"""
        
        # Calcular métricas por ruta para el reporte
        rutas_detalle = []
        capacidad_camion = st.session_state.get('capacidad_camion', 20)
        
        for idx, ruta in enumerate(rutas):
            if len(ruta) > 2:
                distancia = sum(matriz_distancias[ruta[i]][ruta[i+1]] for i in range(len(ruta)-1)) / 1000
                demanda = sum(clientes[node_id].demanda for node_id in ruta if node_id != 0)
                utilizacion = (demanda / capacidad_camion) * 100 if capacidad_camion > 0 else 0
                
                rutas_detalle.append({
                    "vehiculo": idx + 1,
                    "clientes": len(ruta) - 2,
                    "distancia": distancia,
                    "demanda": demanda,
                    "utilizacion": utilizacion
                })
        
        # Generar filas HTML para rutas
        filas_rutas = ""
        for ruta in rutas_detalle:
            filas_rutas += f"""
            <tr>
                <td>{ruta['vehiculo']}</td>
                <td>{ruta['clientes']}</td>
                <td>{ruta['distancia']:.1f}</td>
                <td>{ruta['demanda']}</td>
                <td>{ruta['utilizacion']:.1f}%</td>
            </tr>
            """
        
        # Generar filas HTML para clientes urgentes
        filas_clientes = ""
        for cliente in clientes.values():
            if cliente.id != 0 and cliente.prioridad >= 2:
                filas_clientes += f"""
                <tr>
                    <td>{cliente.nombre}</td>
                    <td>{cliente.tipo}</td>
                    <td>{cliente.demanda}</td>
                    <td>{cliente.ventana_tiempo[0]}-{cliente.ventana_tiempo[1]}</td>
                    <td>{'★' * cliente.prioridad}</td>
                </tr>
                """
        
        reporte = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte Optimización Logística</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; }}
                .kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .kpi-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚚 Reporte de Optimización Logística</h1>
                <p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="kpis">
                <div class="kpi-card">
                    <h3>📊 Resumen</h3>
                    <p><strong>Clientes:</strong> {kpis['total_clientes']}</p>
                    <p><strong>Vehículos:</strong> {kpis['vehiculos_utilizados']}/{kpis['vehiculos_totales']}</p>
                    <p><strong>Distancia Total:</strong> {kpis['distancia_total_km']:.1f} km</p>
                </div>
                
                <div class="kpi-card">
                    <h3>💰 Costos</h3>
                    <p><strong>Costo Total:</strong> ${kpis['costo_total']:.2f}</p>
                    <p><strong>Costo por Entrega:</strong> ${kpis['costo_total']/max(1, kpis['total_clientes']):.2f}</p>
                    <p><strong>Costo por Km:</strong> ${kpis['costo_total']/max(1, kpis['distancia_total_km']):.2f}</p>
                </div>
                
                <div class="kpi-card">
                    <h3>⚡ Eficiencia</h3>
                    <p><strong>Utilización:</strong> {kpis['utilizacion_promedio']:.1f}%</p>
                    <p><strong>OTR:</strong> {kpis['otr_promedio']*100:.1f}%</p>
                    <p><strong>Tiempo Promedio:</strong> {kpis['tiempo_total_horas']/max(1, kpis['vehiculos_utilizados']):.1f} h/vehículo</p>
                </div>
            </div>
            
            <h2>📋 Detalle de Rutas</h2>
            <table>
                <tr>
                    <th>Vehículo</th>
                    <th>Clientes</th>
                    <th>Distancia (km)</th>
                    <th>Demanda</th>
                    <th>Utilización</th>
                </tr>
                {filas_rutas}
            </table>
            
            <h2>📍 Clientes Críticos</h2>
            <table>
                <tr>
                    <th>Cliente</th>
                    <th>Tipo</th>
                    <th>Demanda</th>
                    <th>Ventana</th>
                    <th>Prioridad</th>
                </tr>
                {filas_clientes}
            </table>
            
            <footer style="margin-top: 40px; padding: 20px; text-align: center; color: #666;">
                <p>TMS Logístico Profesional • Reporte generado automáticamente</p>
            </footer>
        </body>
        </html>
        """
        
        return reporte

# ------------------------------------------
# INTERFAZ PRINCIPAL STREAMLIT
# ------------------------------------------
def main():
    """Función principal de la aplicación"""
    
    # Inicializar estado de sesión con valores por defecto
    if 'clientes' not in st.session_state:
        st.session_state.clientes = None
    if 'matriz_distancias' not in st.session_state:
        st.session_state.matriz_distancias = None
    if 'rutas' not in st.session_state:
        st.session_state.rutas = None
    if 'kpis' not in st.session_state:
        st.session_state.kpis = None
    if 'num_clientes' not in st.session_state:
        st.session_state.num_clientes = 50
    if 'num_camiones' not in st.session_state:
        st.session_state.num_camiones = 12
    if 'capacidad_camion' not in st.session_state:
        st.session_state.capacidad_camion = 20
    if 'ruta_seleccionada' not in st.session_state:
        st.session_state.ruta_seleccionada = None
    
    # Barra lateral - Configuración
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <h1>🚚</h1>
            <h3>TMS Logístico</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.title("⚙️ Configuración")
        
        # Parámetros básicos
        st.session_state.num_clientes = st.slider(
            "Número de clientes", 10, 200, 50, 
            help="Cantidad de puntos de entrega"
        )
        
        st.session_state.num_camiones = st.slider(
            "Número de vehículos", 2, 30, 12,
            help="Tamaño de la flota disponible"
        )
        
        st.session_state.capacidad_camion = st.slider(
            "Capacidad por vehículo", 5, 50, 20,
            help="Capacidad máxima de carga por vehículo"
        )
        
        # Estrategia de optimización
        estrategia = st.selectbox(
            "Estrategia de optimización",
            ["Clustering + VRP", "VRP Tradicional", "Balance Perfecto", "Minimizar Costos"],
            index=0
        )
        
        # Botón para generar datos
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Generar", use_container_width=True):
                with st.spinner("Generando datos..."):
                    generador = GeneradorDatos()
                    st.session_state.clientes = generador.generar_clientes(
                        st.session_state.num_clientes
                    )
                    st.session_state.matriz_distancias = generador.generar_matriz_distancias(
                        st.session_state.clientes
                    )
                    # Limpiar rutas anteriores
                    st.session_state.rutas = None
                    st.session_state.kpis = None
                    st.rerun()
        
        with col2:
            if st.button("🚀 Optimizar", use_container_width=True, 
                        disabled=st.session_state.clientes is None):
                with st.spinner("Optimizando rutas..."):
                    optimizador = OptimizadorVRP()
                    st.session_state.rutas = optimizador.optimizar_con_clustering(
                        st.session_state.clientes,
                        st.session_state.matriz_distancias,
                        st.session_state.num_camiones,
                        st.session_state.capacidad_camion
                    )
                    st.session_state.kpis = optimizador.calcular_kpis(
                        st.session_state.rutas,
                        st.session_state.clientes,
                        st.session_state.matriz_distancias
                    )
                    st.rerun()
        
        st.divider()
        
        # Sistema de alertas
        if st.session_state.rutas:
            alertas = SistemaAlertas()
            alertas.verificar_rutas(
                st.session_state.rutas,
                st.session_state.clientes,
                st.session_state.capacidad_camion
            )
            alertas.mostrar_alertas()
        else:
            st.sidebar.info("Optimiza tus rutas")
    
    # Encabezado principal
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">🚚 TMS LOGÍSTICO (Transportation Management System)</h1>
        <h3 style="color: white; margin: 0;">Sistema Inteligente de Optimización de Rutas</h3>
        <p style="color: rgba(255,255,255,0.8);">Managua, Nicaragua • {}</p>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y")), unsafe_allow_html=True)
    
    # Pestañas principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🗺️ Mapa Interactivo", 
        "📈 Análisis", 
        "🔄 Simulaciones",
        "📋 Reportes"
    ])
    
    # Dashboard
    with tab1:
        if st.session_state.kpis:
            st.header("📊 Dashboard Ejecutivo")
            
            # KPIs principales
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.markdown("""
                <div style="text-align: center; padding: 15px; border-radius: 10px; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <div style="font-size: 24px; color: white;">👥</div>
                    <div style="font-size: 28px; font-weight: bold; color: white;">{}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.8);">Clientes Atendidos</div>
                </div>
                """.format(st.session_state.kpis['total_clientes']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 15px; border-radius: 10px; 
                            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <div style="font-size: 24px; color: white;">🚚</div>
                    <div style="font-size: 28px; font-weight: bold; color: white;">{}/{}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.8);">Vehículos Activos</div>
                </div>
                """.format(st.session_state.kpis['vehiculos_utilizados'], 
                        st.session_state.kpis['vehiculos_totales']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="text-align: center; padding: 15px; border-radius: 10px; 
                            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <div style="font-size: 24px; color: white;">💰</div>
                    <div style="font-size: 28px; font-weight: bold; color: white;">${:.2f}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.8);">Costo Total</div>
                </div>
                """.format(st.session_state.kpis['costo_total']), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div style="text-align: center; padding: 15px; border-radius: 10px; 
                            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                    <div style="font-size: 24px; color: white;">⚡</div>
                    <div style="font-size: 28px; font-weight: bold; color: white;">{:.1f}%</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.8);">Utilización</div>
                </div>
                """.format(st.session_state.kpis['utilizacion_promedio']), unsafe_allow_html=True)
            
            with col5:
                st.markdown("""
                <div style="text-align: center; padding: 15px; border-radius: 10px; 
                            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);">
                    <div style="font-size: 24px; color: white;">🛣️</div>
                    <div style="font-size: 28px; font-weight: bold; color: white;">{:.1f} km</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.8);">Distancia Total</div>
                </div>
                """.format(st.session_state.kpis['distancia_total_km']), unsafe_allow_html=True)
            
            with col6:
                st.markdown("""
                <div style="text-align: center; padding: 15px; border-radius: 10px; 
                            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                    <div style="font-size: 24px; color: white;">🎯</div>
                    <div style="font-size: 28px; font-weight: bold; color: white;">{:.1f}%</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.8);">OTR</div>
                </div>
                """.format(st.session_state.kpis['otr_promedio'] * 100), unsafe_allow_html=True)
            
            # Gráficos principales
            visualizador = Visualizador()
            fig_kpis = visualizador.crear_dashboard_kpis(st.session_state.kpis)
            st.plotly_chart(fig_kpis, use_container_width=True)
            
            # Tabla de resumen de rutas
            st.subheader("📋 Resumen de Rutas")
            
            datos_rutas = []
            for idx, ruta in enumerate(st.session_state.rutas):
                if len(ruta) > 2:
                    distancia = sum(st.session_state.matriz_distancias[ruta[i]][ruta[i+1]] 
                                for i in range(len(ruta)-1)) / 1000
                    demanda = sum(st.session_state.clientes[node_id].demanda 
                                for node_id in ruta if node_id != 0)
                    utilizacion = (demanda / st.session_state.capacidad_camion) * 100
                    
                    datos_rutas.append({
                        "Vehículo": idx + 1,
                        "Paradas": len(ruta) - 2,
                        "Distancia (km)": round(distancia, 1),
                        "Demanda": demanda,
                        "Utilización (%)": round(utilizacion, 1),
                        "Tiempo Est. (h)": round(distancia / config["velocidad_promedio"] + 
                                                (len(ruta)-2) * config["tiempo_servicio"] / 60, 1)
                    })
            
            if datos_rutas:
                df_rutas = pd.DataFrame(datos_rutas)
                st.dataframe(
                    df_rutas.style.background_gradient(
                        subset=['Utilización (%)'], 
                        cmap='YlOrRd'
                    ),
                    use_container_width=True,
                    height=300
                )
        else:
            st.info("👈 Configura los parámetros y optimiza las rutas para ver el dashboard")
    
    # Mapa Interactivo
    with tab2:
        if st.session_state.clientes and st.session_state.rutas:
            st.header("🗺️ Mapa Interactivo de Rutas")
            
            # Selector de tipo de mapa
            tipo_mapa = st.radio(
                "Selecciona el tipo de mapa:",
                ["🌍 Mapa General con Folium", "🔥 Mapa de Calor", "📊 Mapa con Clusters"],
                horizontal=True
            )
            
            # Selector de ruta
            rutas_validas = [i for i in range(len(st.session_state.rutas)) 
                           if len(st.session_state.rutas[i]) > 2]
            
            if rutas_validas:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    opciones_rutas = ["Todas las rutas"] + [f"Ruta {i+1}" for i in rutas_validas]
                    ruta_seleccionada_str = st.selectbox(
                        "Ver ruta específica:",
                        opciones_rutas
                    )
                    
                    if ruta_seleccionada_str == "Todas las rutas":
                        st.session_state.ruta_seleccionada = None
                    else:
                        st.session_state.ruta_seleccionada = int(ruta_seleccionada_str.split()[1]) - 1
                
                with col1:
                    mapa_interactivo = MapaInteractivo()
                    
                    try:
                        from streamlit_folium import st_folium
                    except ImportError:
                        st.error("❌ Falta instalar streamlit-folium. Ejecuta: pip install streamlit-folium")
                        st.info("Usando mapa alternativo con Plotly...")
                        visualizador = Visualizador()
                        fig_mapa = visualizador.crear_mapa_calor(
                            st.session_state.clientes,
                            st.session_state.rutas
                        )
                        st.plotly_chart(fig_mapa, use_container_width=True)
                    else:
                        if tipo_mapa == "🌍 Mapa General con Folium":
                            mapa = mapa_interactivo.crear_mapa_folium(
                                st.session_state.clientes,
                                st.session_state.rutas,
                                st.session_state.ruta_seleccionada
                            )
                            
                            if mapa:
                                # Mostrar el mapa con st_folium
                                mapa_data = st_folium(
                                    mapa,
                                    width=1200,
                                    height=600,
                                    returned_objects=[]
                                )
                                
                                st.markdown("""
                                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
                                    <h4>🎯 Controles del Mapa</h4>
                                    <p>• <strong>Click en marcadores:</strong> Ver información detallada del cliente</p>
                                    <p>• <strong>Selector superior derecho:</strong> Cambiar tipo de mapa (calles, satélite, relieve)</p>
                                    <p>• <strong>Control de capas:</strong> Mostrar/ocultar rutas específicas</p>
                                    <p>• <strong>Herramienta de medición:</strong> Icono de regla para medir distancias</p>
                                    <p>• <strong>Pantalla completa:</strong> Icono de expansión en esquina inferior derecha</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        elif tipo_mapa == "🔥 Mapa de Calor":
                            mapa = mapa_interactivo.crear_mapa_calor_densidad(st.session_state.clientes)
                            
                            if mapa:
                                mapa_data = st_folium(
                                    mapa,
                                    width=1200,
                                    height=600,
                                    returned_objects=[]
                                )
                        
                        else:  # Mapa con Clusters
                            mapa = mapa_interactivo.crear_mapa_clusters(
                                st.session_state.clientes,
                                st.session_state.rutas
                            )
                            
                            if mapa:
                                mapa_data = st_folium(
                                    mapa,
                                    width=1200,
                                    height=600,
                                    returned_objects=[]
                                )
                
                # Detalle de la ruta seleccionada
                if st.session_state.ruta_seleccionada is not None:
                    st.subheader(f"📍 Detalle de Ruta {st.session_state.ruta_seleccionada + 1}")
                    
                    ruta_idx = st.session_state.ruta_seleccionada
                    ruta = st.session_state.rutas[ruta_idx]
                    
                    col_detalle1, col_detalle2 = st.columns(2)
                    
                    with col_detalle1:
                        st.markdown("### 🚛 Paradas de la Ruta")
                        for i, cliente_id in enumerate(ruta):
                            if cliente_id == 0:
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid black;">
                                    <strong>🏭 {st.session_state.clientes[cliente_id].nombre}</strong>
                                    <br><small>Punto de partida y llegada</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                cliente = st.session_state.clientes[cliente_id]
                                color_borde = {
                                    'normal': '#4ECDC4',
                                    'urgente': '#FF6B6B',
                                    'empresa': '#45B7D1',
                                    'programado': '#FFEAA7'
                                }.get(cliente.tipo, '#4ECDC4')
                                
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid {color_borde};">
                                    <strong>📦 Parada {i}: {cliente.nombre}</strong>
                                    <br><small>Demanda: {cliente.demanda} unidades • Tipo: {cliente.tipo}</small>
                                    <br><small>Ventana: {cliente.ventana_tiempo[0]//60}:{str(cliente.ventana_tiempo[0]%60).zfill(2)} - {cliente.ventana_tiempo[1]//60}:{str(cliente.ventana_tiempo[1]%60).zfill(2)}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with col_detalle2:
                        # Métricas de la ruta seleccionada
                        distancia = sum(st.session_state.matriz_distancias[ruta[i]][ruta[i+1]] 
                                      for i in range(len(ruta)-1)) / 1000
                        demanda = sum(st.session_state.clientes[node_id].demanda 
                                    for node_id in ruta if node_id != 0)
                        tiempo_viaje = distancia / config["velocidad_promedio"]
                        tiempo_servicio = (len(ruta)-2) * config["tiempo_servicio"] / 60
                        tiempo_total = tiempo_viaje + tiempo_servicio
                        costo = (distancia * config["costo_por_km"]) + (tiempo_total * config["costo_por_hora"])
                        
                        st.metric("📏 Distancia Total", f"{distancia:.1f} km")
                        st.metric("📦 Demanda Total", f"{demanda} unidades")
                        st.metric("⏱️ Tiempo Estimado", f"{tiempo_total:.1f} horas")
                        st.metric("📊 Utilización", f"{(demanda/st.session_state.capacidad_camion)*100:.1f}%")
                        st.metric("💰 Costo Estimado", f"${costo:.2f}")
                        
                        # Progreso de la ruta
                        st.markdown("### 📈 Progreso de la Ruta")
                        
                        # Calcular tiempos acumulados
                        tiempos_acumulados = [0]
                        tiempo_acumulado = 0
                        
                        for i in range(len(ruta) - 1):
                            distancia_segmento = st.session_state.matriz_distancias[ruta[i]][ruta[i+1]] / 1000
                            tiempo_segmento = distancia_segmento / config["velocidad_promedio"]
                            
                            if ruta[i] != 0:  # Tiempo de servicio en cliente actual
                                tiempo_acumulado += config["tiempo_servicio"] / 60
                            
                            tiempo_acumulado += tiempo_segmento
                            tiempos_acumulados.append(tiempo_acumulado)
                        
                        # Crear gráfico de progreso
                        datos_progreso = pd.DataFrame({
                            'Parada': [f'P{i}' for i in range(len(ruta))],
                            'Tiempo Acumulado (h)': tiempos_acumulados,
                            'Demanda': [st.session_state.clientes[node_id].demanda if node_id != 0 else 0 for node_id in ruta]
                        })
                        
                        fig_progreso = px.line(
                            datos_progreso,
                            x='Parada',
                            y='Tiempo Acumulado (h)',
                            title=f'Progreso Temporal - Ruta {ruta_idx + 1}',
                            markers=True,
                            height=300
                        )
                        
                        fig_progreso.update_traces(
                            line=dict(color=mapa_interactivo.colores_rutas[ruta_idx % len(mapa_interactivo.colores_rutas)], width=3)
                        )
                        
                        fig_progreso.update_layout(
                            xaxis_title="Parada",
                            yaxis_title="Horas Acumuladas",
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_progreso, use_container_width=True)
            else:
                st.info("No hay rutas válidas para mostrar")
        else:
            st.info("👈 Genera y optimiza rutas para ver el mapa interactivo")
    
    # Análisis
    with tab3:
        if st.session_state.kpis:
            st.header("📈 Análisis Avanzado")
            
            visualizador = Visualizador()
            
            # Seleccionar gráfico
            tipo_grafico = st.selectbox(
                "Seleccionar tipo de análisis:",
                ["Distribución de Tiempos", "Perfil de Desempeño", "Flujo de Recursos", "Matriz de Desempeño"]
            )
            
            if tipo_grafico == "Distribución de Tiempos":
                st.subheader("Distribución de Tiempos por Vehículo")
                fig = visualizador.crear_grafico_distribucion_tiempos(
                    st.session_state.rutas,
                    st.session_state.clientes,
                    st.session_state.matriz_distancias
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif tipo_grafico == "Perfil de Desempeño":
                st.subheader("📡 Perfil de Desempeño por Vehículo")
                fig = visualizador.crear_grafico_radar_desempeno(
                    st.session_state.rutas,
                    st.session_state.clientes,
                    st.session_state.matriz_distancias
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif tipo_grafico == "Flujo de Recursos":
                st.subheader("🌊 Flujo de Recursos y Tiempos")
                fig = visualizador.crear_grafico_sankey_recursos(
                    st.session_state.rutas,
                    st.session_state.clientes,
                    st.session_state.matriz_distancias
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Matriz de Desempeño
                st.subheader("📊 Matriz de Desempeño por Vehículo")
                fig = visualizador.crear_grafico_burbujas_desempeno(
                    st.session_state.rutas,
                    st.session_state.clientes,
                    st.session_state.matriz_distancias
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de costos
            st.subheader("💰 Análisis de Costos")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                costo_combustible = (st.session_state.kpis['distancia_total_km'] * 
                                   config["costo_por_km"])
                st.metric("Costo Combustible", f"${costo_combustible:.2f}")
            
            with col2:
                costo_tiempo = (st.session_state.kpis['tiempo_total_horas'] * 
                              config["costo_por_hora"])
                st.metric("Costo Tiempo", f"${costo_tiempo:.2f}")
            
            with col3:
                costo_total = st.session_state.kpis['costo_total']
                st.metric("Costo Total", f"${costo_total:.2f}")
            
            # Distribución de tipos de cliente
            st.subheader("👥 Distribución de Clientes")
            fig_tipos = visualizador.crear_grafico_distribucion_clientes(st.session_state.clientes)
            st.plotly_chart(fig_tipos, use_container_width=True)
            
        else:
            st.info("👈 Optimiza rutas para ver análisis detallados")
    
    # Simulaciones
    with tab4:
        st.header("🔄 Simulador What-If")
        
        if st.session_state.kpis:
            simulador = SimuladorWhatIf()
            
            escenario = st.selectbox(
                "Seleccionar escenario a simular",
                list(simulador.escenarios.keys()),
                format_func=lambda x: simulador.escenarios[x]
            )
            
            if st.button("🚀 Ejecutar Simulación", use_container_width=True):
                with st.spinner("Simulando escenario..."):
                    kpis_simulados = simulador.simular_escenario(
                        escenario,
                        st.session_state.kpis,
                        st.session_state.clientes
                    )
                    
                    simulador.mostrar_comparativa(
                        st.session_state.kpis,
                        kpis_simulados,
                        simulador.escenarios[escenario]
                    )
                    
                    # Mostrar impactos detallados
                    st.subheader("📊 Impacto Detallado")
                    
                    impactos = pd.DataFrame({
                        "Métrica": ["Costo Total", "Tiempo Total", "Utilización", "OTR"],
                        "Base": [
                            st.session_state.kpis['costo_total'],
                            st.session_state.kpis['tiempo_total_horas'],
                            st.session_state.kpis['utilizacion_promedio'],
                            st.session_state.kpis['otr_promedio'] * 100
                        ],
                        "Simulado": [
                            kpis_simulados['costo_total'],
                            kpis_simulados['tiempo_total_horas'],
                            kpis_simulados['utilizacion_promedio'],
                            kpis_simulados['otr_promedio'] * 100
                        ],
                        "Cambio %": [
                            ((kpis_simulados['costo_total'] - st.session_state.kpis['costo_total']) / 
                             st.session_state.kpis['costo_total'] * 100) if st.session_state.kpis['costo_total'] > 0 else 0,
                            ((kpis_simulados['tiempo_total_horas'] - st.session_state.kpis['tiempo_total_horas']) / 
                             st.session_state.kpis['tiempo_total_horas'] * 100) if st.session_state.kpis['tiempo_total_horas'] > 0 else 0,
                            ((kpis_simulados['utilizacion_promedio'] - st.session_state.kpis['utilizacion_promedio']) / 
                             st.session_state.kpis['utilizacion_promedio'] * 100) if st.session_state.kpis['utilizacion_promedio'] > 0 else 0,
                            ((kpis_simulados['otr_promedio'] - st.session_state.kpis['otr_promedio']) / 
                             st.session_state.kpis['otr_promedio'] * 100) if st.session_state.kpis['otr_promedio'] > 0 else 0
                        ]
                    })
                    
                    st.dataframe(
                        impactos.style.format({
                            'Base': '${:.2f}' if impactos['Métrica'][0] == 'Costo Total' else '{:.1f}',
                            'Simulado': '${:.2f}' if impactos['Métrica'][0] == 'Costo Total' else '{:.1f}',
                            'Cambio %': '{:+.1f}%'
                        }).background_gradient(
                            subset=['Cambio %'], 
                            cmap='RdYlGn', 
                            vmin=-50, 
                            vmax=50
                        ),
                        use_container_width=True
                    )
        else:
            st.info("👈 Optimiza rutas primero para ejecutar simulaciones")
    
    # Reportes
    with tab5:
        st.header("📋 Sistema de Reportes")
        
        if st.session_state.clientes and st.session_state.rutas and st.session_state.kpis:
            sistema_exportacion = SistemaExportacion()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                
                if st.button("📄 Generar Reporte HTML", use_container_width=True):
                    reporte_html = sistema_exportacion.generar_reporte_html(
                        st.session_state.clientes,
                        st.session_state.rutas,
                        st.session_state.kpis,
                        st.session_state.matriz_distancias
                    )
                    
                    with open("reporte_optimizacion.html", "w", encoding="utf-8") as f:
                        f.write(reporte_html)
                    
                    with open("reporte_optimizacion.html", "r", encoding="utf-8") as f:
                        st.download_button(
                            label="⬇️ Descargar HTML",
                            data=f,
                            file_name="reporte_optimizacion.html",
                            mime="text/html",
                            use_container_width=True
                        )
            
            with col2:
                if st.button("📋 Copiar Resumen", use_container_width=True):
                    resumen = f"""
                    📊 RESUMEN DE OPTIMIZACIÓN
                    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    
                    📈 KPIs PRINCIPALES:
                    • Clientes atendidos: {st.session_state.kpis['total_clientes']}
                    • Vehículos utilizados: {st.session_state.kpis['vehiculos_utilizados']}/{st.session_state.kpis['vehiculos_totales']}
                    • Distancia total: {st.session_state.kpis['distancia_total_km']:.1f} km
                    • Costo total: ${st.session_state.kpis['costo_total']:.2f}
                    • Utilización promedio: {st.session_state.kpis['utilizacion_promedio']:.1f}%
                    • OTR: {st.session_state.kpis['otr_promedio']*100:.1f}%
                    
                    🎯 RECOMENDACIONES:
                    """
                    
                    st.code(resumen, language="text")
                    st.info("Resumen copiado al portapapeles")
            
            # Vista previa de datos
            st.subheader("👁️ Vista Previa de Datos")
            
            tab_clientes, tab_rutas, tab_kpis = st.tabs(["Clientes", "Rutas", "KPIs"])
            
            with tab_clientes:
                st.dataframe(
                    pd.DataFrame([c.to_dict() for c in st.session_state.clientes.values()]),
                    use_container_width=True,
                    height=300
                )
            
            with tab_rutas:
                datos_rutas = []
                for idx, ruta in enumerate(st.session_state.rutas):
                    if len(ruta) > 2:
                        datos_rutas.append({
                            "Vehículo": idx + 1,
                            "Paradas": len(ruta) - 2,
                            "Secuencia": " → ".join([st.session_state.clientes[node_id].nombre[:10] 
                                                    for node_id in ruta])
                        })
                
                if datos_rutas:
                    st.dataframe(pd.DataFrame(datos_rutas), use_container_width=True)
            
            with tab_kpis:
                st.json(st.session_state.kpis)
        else:
            st.info("👈 Genera y optimiza rutas para acceder a los reportes")
    
    # Pie de página
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>🚚 TMS Logístico Profesional</strong> • Sistema de optimización de rutas con mapas interactivos</p>
        <p>Managua, Nicaragua • © 2024 • Desarrollado con Streamlit y Folium</p>
        <p><small>📧 ivanb.samuel@gmail.com • 📞 +505 8855 9683</small></p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# EJECUTA LA APLICACIÓN
# ------------------------------------------
if __name__ == "__main__":
    main()