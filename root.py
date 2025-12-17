# archivo: rutas_dijkstra_managua.py

import osmnx as ox
import networkx as nx
import folium
import streamlit as st
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
from math import radians, cos

# Configuración de la página
st.set_page_config(page_title="Rutas Managua", page_icon="🗺️", layout="wide")

# Inicializar estado de sesión
if 'ruta_calculada' not in st.session_state:
    st.session_state.ruta_calculada = False
    st.session_state.mapa = None
    st.session_state.distancia_total = 0
    st.session_state.tiempo_minutos = 0
    st.session_state.ruta_corta = []
    st.session_state.coords = []

# Título de la app
st.title("🚗 Optimización de rutas en Managua con Dijkstra")
st.markdown("---")

# Cache para el grafo
@st.cache_data
def cargar_grafo():
    st.info("📥 Descargando mapa de Managua por primera vez... Esto puede tardar unos segundos.")
    try:
        G = ox.graph_from_place('Managua, Nicaragua', network_type='drive')
        G = G.to_undirected()
        return G
    except Exception as e:
        st.error(f"Error al cargar el mapa: {e}")
        return None

# Cargar grafo
with st.spinner("Cargando mapa de Managua..."):
    G = cargar_grafo()

if G is None:
    st.stop()

# Función para calcular área aproximada
def calcular_area_aproximada(G):
    try:
        lats = [data['y'] for _, data in G.nodes(data=True)]
        lons = [data['x'] for _, data in G.nodes(data=True)]
        
        if not lats or not lons:
            return 0
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        lat_media = radians((min_lat + max_lat) / 2)
        altura_km = (max_lat - min_lat) * 111
        ancho_km = (max_lon - min_lon) * 111 * cos(lat_media)
        
        area_km2 = altura_km * ancho_km
        return max(0, area_km2)
    except:
        return 0

# Información del grafo
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nodos", len(G.nodes()))
with col2:
    st.metric("Calles", len(G.edges()))
with col3:
    area = calcular_area_aproximada(G)
    st.metric("Área (km²)", f"{area:.1f}")

st.markdown("---")

# Selección de puntos de origen y destino
st.subheader("📍 Selecciona origen y destino")

# Función para geocodificar direcciones
@st.cache_data
def geocodificar(direccion):
    try:
        geolocator = Nominatim(user_agent="managua_routes")
        location = geolocator.geocode(direccion + ", Managua, Nicaragua")
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None

# Dos columnas para origen y destino
col_origen, col_destino = st.columns(2)

nodos_lista = list(G.nodes())[:1000]

with col_origen:
    st.markdown("### Origen")
    origen_tipo = st.radio("Tipo de selección:", ["Nodo", "Dirección"], key="origen", horizontal=True)
    
    if origen_tipo == "Dirección":
        dir_origen = st.text_input("Ingresa dirección:", "Universidad Nacional de Ingeniería, Managua", key="dir_origen")
        if dir_origen:
            coord = geocodificar(dir_origen)
            if coord:
                origen = ox.distance.nearest_nodes(G, coord[1], coord[0])
                lat_origen = G.nodes[origen]['y']
                lon_origen = G.nodes[origen]['x']
                st.success(f"Ubicado: {lat_origen:.6f}, {lon_origen:.6f}")
            else:
                st.warning("Dirección no encontrada.")
                origen = nodos_lista[0]
                lat_origen = G.nodes[origen]['y']
                lon_origen = G.nodes[origen]['x']
        else:
            origen = nodos_lista[0]
            lat_origen = G.nodes[origen]['y']
            lon_origen = G.nodes[origen]['x']
    else:
        origen_idx = st.selectbox("Selecciona nodo origen:", 
                                 range(len(nodos_lista)),
                                 format_func=lambda i: f"Nodo {nodos_lista[i]}", 
                                 key="origen_select")
        origen = nodos_lista[origen_idx]
        lat_origen = G.nodes[origen]['y']
        lon_origen = G.nodes[origen]['x']

with col_destino:
    st.markdown("### Destino")
    destino_tipo = st.radio("Tipo de selección:", ["Nodo", "Dirección"], key="destino", horizontal=True)
    
    if destino_tipo == "Dirección":
        dir_destino = st.text_input("Ingresa dirección:", "Metrocentro Managua", key="dir_destino")
        if dir_destino:
            coord = geocodificar(dir_destino)
            if coord:
                destino = ox.distance.nearest_nodes(G, coord[1], coord[0])
                lat_destino = G.nodes[destino]['y']
                lon_destino = G.nodes[destino]['x']
                st.success(f"Ubicado: {lat_destino:.6f}, {lon_destino:.6f}")
            else:
                st.warning("Dirección no encontrada.")
                destino = nodos_lista[-1]
                lat_destino = G.nodes[destino]['y']
                lon_destino = G.nodes[destino]['x']
        else:
            destino = nodos_lista[-1]
            lat_destino = G.nodes[destino]['y']
            lon_destino = G.nodes[destino]['x']
    else:
        destino_idx = st.selectbox("Selecciona nodo destino:", 
                                  range(len(nodos_lista)),
                                  format_func=lambda i: f"Nodo {nodos_lista[i]}", 
                                  index=min(100, len(nodos_lista)-1),
                                  key="destino_select")
        destino = nodos_lista[destino_idx]
        lat_destino = G.nodes[destino]['y']
        lon_destino = G.nodes[destino]['x']

# Mostrar coordenadas
col_coords1, col_coords2 = st.columns(2)
with col_coords1:
    st.info(f"**Origen:** ({lat_origen:.6f}, {lon_origen:.6f})")

with col_coords2:
    st.info(f"**Destino:** ({lat_destino:.6f}, {lon_destino:.6f})")

# Calcular distancia en línea recta
distancia_linea_recta = ox.distance.great_circle(lat_origen, lon_origen, lat_destino, lon_destino)
st.info(f"📐 **Distancia en línea recta:** {distancia_linea_recta:.2f} km")

st.markdown("---")

# Botón para calcular ruta
if st.button("🚀 Calcular Ruta Óptima", type="primary", use_container_width=True):
    
    with st.spinner("Calculando ruta más corta..."):
        try:
            # Calcular ruta usando Dijkstra
            ruta_corta = nx.dijkstra_path(G, origen, destino, weight='length')
            distancia_total = nx.dijkstra_path_length(G, origen, destino, weight='length')
            
            # Calcular tiempo estimado
            tiempo_minutos = (distancia_total / 1000) / 30 * 60
            
            # Obtener coordenadas para el mapa
            coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in ruta_corta]
            
            # Crear mapa centrado entre origen y destino
            lat_centro = (lat_origen + lat_destino) / 2
            lon_centro = (lon_origen + lon_destino) / 2
            
            m = folium.Map(location=[lat_centro, lon_centro], zoom_start=13)
            
            # Dibujar ruta
            folium.PolyLine(
                coords, 
                color='blue', 
                weight=5, 
                opacity=0.8,
                tooltip=f"Distancia: {distancia_total/1000:.2f} km"
            ).add_to(m)
            
            # Marcadores
            folium.Marker(
                location=coords[0], 
                popup="Origen",
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(m)
            
            folium.Marker(
                location=coords[-1], 
                popup="Destino",
                icon=folium.Icon(color='red', icon='stop', prefix='fa')
            ).add_to(m)
            
            # Guardar en estado de sesión
            st.session_state.ruta_calculada = True
            st.session_state.mapa = m
            st.session_state.distancia_total = distancia_total
            st.session_state.tiempo_minutos = tiempo_minutos
            st.session_state.ruta_corta = ruta_corta
            st.session_state.coords = coords
            st.session_state.lat_origen = lat_origen
            st.session_state.lon_origen = lon_origen
            st.session_state.lat_destino = lat_destino
            st.session_state.lon_destino = lon_destino
            
        except nx.NetworkXNoPath:
            st.error("❌ No se encontró una ruta entre los puntos seleccionados.")
            st.session_state.ruta_calculada = False
        except Exception as e:
            st.error(f"❌ Error al calcular la ruta: {str(e)}")
            st.session_state.ruta_calculada = False

# Mostrar resultados si hay una ruta calculada
if st.session_state.ruta_calculada and st.session_state.mapa is not None:
    st.markdown("---")
    st.success(f"✅ **Ruta calculada exitosamente!**")
    
    # Mostrar resultados en columnas
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("📏 Distancia Total", f"{st.session_state.distancia_total/1000:.2f} km")
    with col_res2:
        st.metric("⏱️ Tiempo Estimado", f"{st.session_state.tiempo_minutos:.1f} minutos")
    with col_res3:
        st.metric("🛣️ Segmentos", len(st.session_state.ruta_corta))
    
    # Mostrar eficiencia de la ruta
    eficiencia = (distancia_linea_recta * 1000) / st.session_state.distancia_total * 100
    st.progress(min(100, int(eficiencia)))
    st.caption(f"📊 **Eficiencia de la ruta:** {eficiencia:.1f}% (vs. línea recta)")
    
    st.markdown("---")
    
    # Mostrar mapa con st_folium
    st.subheader("🗺️ Mapa de la Ruta")
    
    # Mostrar el mapa usando st_folium
    mapa_data = st_folium(st.session_state.mapa, width=900, height=500, key="mapa_ruta")
    
    # Información adicional
    with st.expander("📊 Detalles técnicos"):
        st.write(f"**Nodo origen:** {origen}")
        st.write(f"**Nodo destino:** {destino}")
        st.write(f"**Nodos en la ruta:** {len(st.session_state.ruta_corta)}")
        
        # Mostrar primeros y últimos nodos
        if len(st.session_state.ruta_corta) > 10:
            primeros = st.session_state.ruta_corta[:5]
            ultimos = st.session_state.ruta_corta[-5:]
            st.write("**Primeros 5 nodos:**")
            for nodo in primeros:
                st.write(f"  - Nodo {nodo}: ({G.nodes[nodo]['y']:.6f}, {G.nodes[nodo]['x']:.6f})")
            st.write("...")
            st.write("**Últimos 5 nodos:**")
            for nodo in ultimos:
                st.write(f"  - Nodo {nodo}: ({G.nodes[nodo]['y']:.6f}, {G.nodes[nodo]['x']:.6f})")
        else:
            st.write("**Nodos de la ruta:**")
            for i, nodo in enumerate(st.session_state.ruta_corta):
                st.write(f"  {i+1}. Nodo {nodo}: ({G.nodes[nodo]['y']:.6f}, {G.nodes[nodo]['x']:.6f})")

# Sección de información
with st.expander("ℹ️ Acerca de esta aplicación"):
    st.markdown("""
    ### Características:
    - **Algoritmo Dijkstra**: Calcula la ruta más corta basada en distancia
    - **OpenStreetMap**: Datos actualizados de las calles de Managua
    - **Interfaz intuitiva**: Selección por dirección o nodo
    
    ### Instrucciones:
    1. Selecciona origen y destino (puedes usar direcciones o nodos)
    2. Haz clic en "Calcular Ruta Óptima"
    3. Explora la ruta en el mapa interactivo
    
    ### Dependencias necesarias:
    ```bash
    pip install streamlit osmnx networkx folium geopy pandas numpy
    ```
    
    ### Tecnologías utilizadas:
    - Streamlit para la interfaz web
    - OSMnx para procesamiento de mapas
    - NetworkX para algoritmos de grafos
    - Folium para visualización de mapas
    - Geopy para geocodificación de direcciones
    """)

# Nota al pie
st.markdown("---")
st.caption("© 2024 - Optimización de rutas en Managua | Datos: OpenStreetMap")