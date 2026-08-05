import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import gc
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# =========================================================================
# CONFIGURACIÓN GENERAL Y ESTADO DE SESIÓN
# =========================================================================
st.set_page_config(page_title="Sistema FDA v4 - Modular", layout="wide")
st.title("🔬 Plataforma Integral de Análisis FDA (Sensor de Suelo)")

# Inicializar estado persistente
if "exp_id" not in st.session_state:
    st.session_state.exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Mensaje de cabecera
st.caption(f"🆔 **ID de Sesión Activa:** `{st.session_state.exp_id}`")

# =========================================================================
# NAVEGACIÓN MODULAR (6 TABS)
# =========================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 1. Registro & Ensayo",
    "🌱 2. Preprocesamiento Suelo",
    "📸 3. Captura & Cámara",
    "🎯 4. Segmentación ROIs",
    "📈 5. Cinética & Derivadas",
    "📄 6. Exportar & PDF"
])

# -------------------------------------------------------------------------
# MÓDULO 1: REGISTRO Y METADATOS DEL EXPERIMENTO
# -------------------------------------------------------------------------
with tab1:
    st.header("📋 Módulo 1: Registro del Experimento")
    
    # Selector de Modo de Trabajo
    modo_trabajo = st.radio("Acción a realizar:", ["➕ Crear Nuevo Ensayo", "🔍 Consultar Ensayo Existente (Histórico)"], horizontal=True)
    
    if modo_trabajo == "🔍 Consultar Ensayo Existente (Histórico)":
        st.info("ℹ️ Función de búsqueda habilitada: Los ensayos guardados en Google Sheets aparecerán acá.")
        st.selectbox("Seleccione un Ensayo Previor:", ["EXP-20260805-185212 (Ensayo_Demo)"])
    else:
        st.markdown("Ingrese los metadatos generales para la trazabilidad del análisis.")
        
        col1, col2 = st.columns(2)
        with col1:
            fecha_exp = st.date_input("Fecha del Experimento:", value=date.today())
            operador = st.text_input("Persona a Cargo / Operador:", value="Operador_1")
            nombre_exp = st.text_input("Nombre del Experimento:", value="Ensayo_FDA_Suelo")
            objetivo = st.text_area("Objetivo del Ensayo:", value="Medición de actividad enzimática de hidrólisis de FDA.")
            
            st.subheader("🌡️ Condiciones Ambientales")
            humedad_tierra = st.number_input("Humedad de la tierra (%):", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
            temp_ambiente = st.number_input("Temperatura Ambiente (°C):", value=22.0, step=0.5)
            temp_reaccion = st.number_input("Temperatura en el lugar de Reacción (°C):", value=25.0, step=0.5)

        with col2:
            st.subheader("⚙️ Secuencia Metodológica de Reacción")
            inicio_reaccion = st.selectbox("La reacción se larga adicionando:", ["Tierra", "FDA"])
            
            preincubacion = st.checkbox("Pre-incubación (Buffer con tierra sin FDA)")
            
            # Lógica dinámica para pre-incubación
            if inicio_reaccion == "Tierra":
                tiempo_preinc = 0
                st.info("ℹ️ Al largar la reacción adicionando Tierra, el tiempo de pre-incubación se fija en 0 min.")
            else:
                if preincubacion:
                    tiempo_preinc = st.number_input("Tiempo de pre-incubación (minutos):", min_value=1, value=15)
                else:
                    tiempo_preinc = 0
            
            comentarios_m1 = st.text_area("Comentarios del Módulo 1:", value="")

        st.session_state.meta_m1 = {
            "exp_id": st.session_state.exp_id,
            "fecha": fecha_exp.strftime('%Y-%m-%d'),
            "operador": operador,
            "nombre_exp": nombre_exp,
            "objetivo": objetivo,
            "humedad_tierra": humedad_tierra,
            "temp_ambiente": temp_ambiente,
            "temp_reaccion": temp_reaccion,
            "inicio_reaccion": inicio_reaccion,
            "preincubacion": preincubacion,
            "tiempo_preinc": tiempo_preinc,
            "comentarios": comentarios_m1
        }
        st.success("✅ Metadatos iniciales guardados.")

# -------------------------------------------------------------------------
# MÓDULO 2: PREPROCESAMIENTO DE MUESTRAS DE SUELO
# -------------------------------------------------------------------------
with tab2:
    st.header("🌱 Módulo 2: Preprocesamiento de Suelo")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tamizado = st.checkbox("¿Suelo Tamizado?", value=True)
        malla_tamiz = st.text_input("Apertura de Malla (ej: < 2mm):", value="< 2 mm") if tamizado else "N/A"
        
        # Desplegable con opción "Otro"
        opciones_secado = ["Secado al Aire", "Fresco / Humedad de Campo", "Secado en Estufa (30-40°C)", "Otro"]
        secado_sel = st.selectbox("Estado de Humedad / Secado:", opciones_secado)
        if secado_sel == "Otro":
            secado_detalle = st.text_input("Especifique estado de humedad/secado:")
        else:
            secado_detalle = secado_sel

    with col_s2:
        pesado_preciso = st.checkbox("Muestras pesadas en balanza analítica", value=True)
        obs_muestras = st.text_area("Observaciones sobre el suelo / muestra:", value="Suelo agrícola, textura franco-arcillosa.")
        comentarios_m2 = st.text_area("Comentarios adicionales del Módulo 2:", value="")
        
    st.session_state.meta_m2 = {
        "tamizado": tamizado,
        "malla": malla_tamiz,
        "secado": secado_detalle,
        "observaciones": obs_muestras,
        "comentarios": comentarios_m2
    }

# -------------------------------------------------------------------------
# MÓDULO 3: CARGA DE FOTOS Y METADATOS DE CÁMARA
# -------------------------------------------------------------------------
with tab3:
    st.header("📸 Módulo 3: Carga de Fotos y Parámetros del Dispositivo")
    
    with st.expander("📷 Registro de Parámetros de Cámara / Smartphone", expanded=True):
        c_cam1, c_cam2 = st.columns(2)
        
        with c_cam1:
            # Modelo Dispositivo
            modelos_opt = ["Samsung Galaxy S22", "Samsung Galaxy S21", "Xiaomi Redmi Note", "Motorola Edge", "Otro"]
            mod_sel = st.selectbox("Modelo de Smartphone / Cámara:", modelos_opt)
            modelo_dispositivo = st.text_input("Especifique modelo:") if mod_sel == "Otro" else mod_sel
            
            # ISO
            iso_val = st.text_input("ISO / Sensibilidad:", value="100")
            
            # Apertura
            aperturas_opt = ["1/10", "1/30", "1/60", "1/100", "Otro"]
            ap_sel = st.selectbox("Apertura / Tiempo de Exposición:", aperturas_opt)
            apertura_val = st.text_input("Especifique apertura/exposición:") if ap_sel == "Otro" else ap_sel

        with c_cam2:
            # WB
            wb_opt = ["5500K (Soleado)", "Incandescente", "Fluorescente", "Automático", "Otro"]
            wb_sel = st.selectbox("Balance de Blancos (WB):", wb_opt)
            wb_val = st.text_input("Especifique Balance de Blancos:") if wb_sel == "Otro" else wb_sel
            
            # Enfoque
            enfoque_val = st.text_input("Modo de Enfoque:", value="Manual / Infinito")

    st.subheader("Subida de Secuencia Temporal")
    
    fuente_origen = st.radio("Fuente de las imágenes:", ["📁 Almacenamiento Local / Móvil", "☁️ Google Drive (Carpeta Compartida)"], horizontal=True)
    
    if fuente_origen == "📁 Almacenamiento Local / Móvil":
        archivos_subidos = st.file_uploader("Arrastrá las imágenes (.jpg, .png) desde tu equipo o móvil", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if archivos_subidos:
            st.session_state.archivos_subidos = sorted(archivos_subidos, key=lambda x: x.name)
            st.info(f"📁 {len(st.session_state.archivos_subidos)} imágenes cargadas correctamente.")
    else:
        st.text_input("Ingrese ID de la carpeta compartida en Google Drive:")
        st.info("☁️ Conexión con Drive lista para vincular vía API.")

    if "archivos_subidos" in st.session_state and st.session_state.archivos_subidos:
        tiene_t0 = st.checkbox("¿La primera foto corresponde a un Blanco / Control previo a la reacción (t=0)?")
        st.session_state.tiene_t0 = tiene_t0

# -------------------------------------------------------------------------
# MÓDULOS 4, 5 Y 6 (MAQUETADO DE CONTINUIDAD)
# -------------------------------------------------------------------------
with tab4:
    st.header("🎯 Módulo 4: Detección Automática de ROIs y Mapeo")
    st.info("Segmentación espacial oculta con asignación de gradilla.")

with tab5:
    st.header("📈 Módulo 5: Análisis Cinético y Tasas de Reacción")
    t_corte = st.slider("⏳ Tiempo de inicio de análisis (Ignorar sedimentación inicial en min):", 0.0, 10.0, 5.0, step=0.5)

with tab6:
    st.header("📄 Módulo 6: Descarga de Resultados y Registro Centralizado")
    st.write("Módulo final de exportación a CSV, generación de PDF y sync con Google Sheets.")
