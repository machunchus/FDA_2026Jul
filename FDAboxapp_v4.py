import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import gc
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# =========================================================================
# CONFIGURACIÓN GENERAL Y ESTADO DE SESIÓN
# =========================================================================
st.set_page_config(page_title="Sistema FDA v4 - Modular", layout="wide")
st.title("🔬 Plataforma Integral de Análisis FDA (Sensor de Suelo)")

# Inicializar estructura de estado persistente
if "exp_id" not in st.session_state:
    st.session_state.exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if "procesado" not in st.session_state:
    st.session_state.procesado = False
if "df_export" not in st.session_state:
    st.session_state.df_export = None

# Mensaje de cabecera con ID automático
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
    st.markdown("Ingrese los metadatos generales para la trazabilidad del análisis.")
    
    col1, col2 = st.columns(2)
    with col1:
        operador = st.text_input("Nombre del Operador / Investigador:", value="Operador_1")
        nombre_exp = st.text_input("Nombre del Experimento:", value="Ensayo_FDA_Suelo")
        objetivo = st.text_area("Objetivo del Ensayo:", value="Medición de actividad enzimática de hidrólisis de FDA.")
    
    with col2:
        st.markdown("**Secuencia Metodológica de Reacción:**")
        preincubacion = st.checkbox("¿Se realizó pre-incubación del suelo solo con Buffer?")
        tiempo_preinc = 0
        if preincubacion:
            tiempo_preinc = st.number_input("Tiempo de pre-incubación (minutos):", min_value=1, value=15)
        
        mezcla_directa = st.checkbox("Mezcla directa (Suelo + Buffer + Reactivo FDA desde t=0)", value=not preincubacion)
        
    st.session_state.meta_m1 = {
        "exp_id": st.session_state.exp_id,
        "operador": operador,
        "nombre_exp": nombre_exp,
        "objetivo": objetivo,
        "preincubacion": preincubacion,
        "tiempo_preinc": tiempo_preinc
    }
    st.success("✅ Metadatos iniciales guardados en memoria.")

# -------------------------------------------------------------------------
# MÓDULO 2: PREPROCESAMIENTO DE MUESTRAS DE SUELO
# -------------------------------------------------------------------------
with tab2:
    st.header("🌱 Módulo 2: Preprocesamiento de Suelo")
    st.markdown("Condiciones de acondicionamiento de la matriz.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tamizado = st.checkbox("¿Suelo Tamizado?", value=True)
        malla_tamiz = st.text_input("Apertura de Malla (ej: < 2mm):", value="< 2 mm") if tamizado else "N/A"
        secado = st.selectbox("Estado de Humedad / Secado:", ["Secado al Aire", "Fresco / Humedad de Campo", "Secado en Estufa (30-40°C)"])
    
    with col_s2:
        pesado_preciso = st.checkbox("Muestras pesadas en balanza analítica", value=True)
        obs_muestras = st.text_area("Observaciones sobre el suelo / muestra:", value="Suelo agrícola, textura franco-arcillosa.")
        
    st.session_state.meta_m2 = {
        "tamizado": tamizado,
        "malla": malla_tamiz,
        "secado": secado,
        "observaciones": obs_muestras
    }

# -------------------------------------------------------------------------
# MÓDULO 3: CARGA DE FOTOS Y METADATOS DE CÁMARA
# -------------------------------------------------------------------------
with tab3:
    st.header("📸 Módulo 3: Carga de Fotos y Parámetros del Dispositivo")
    
    with st.expander("📷 Registro de Parámetros de Cámara / Smartphone", expanded=False):
        c_cam1, c_cam2, c_cam3 = st.columns(3)
        modelo_dispositivo = c_cam1.text_input("Modelo de Smartphone / Cámara:", value="Samsung S22 / Camera FV-5")
        iso_val = c_cam2.text_input("ISO / Sensibilidad:", value="100")
        wb_val = c_cam3.text_input("Balance de Blancos (WB):", value="5500K / Soleado")
    
    st.subheader("Subida de Secuencia Temporal")
    archivos_subidos = st.file_uploader("Arrastrá las imágenes (.jpg, .png) desde tu equipo o móvil", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if archivos_subidos:
        st.session_state.archivos_subidos = sorted(archivos_subidos, key=lambda x: x.name)
        st.info(f"📁 {len(st.session_state.archivos_subidos)} imágenes cargadas correctamente.")
        
        tiene_t0 = st.checkbox("¿La primera foto corresponde a un Blanco / Control previo a la reacción (t=0)?")
        st.session_state.tiene_t0 = tiene_t0

# -------------------------------------------------------------------------
# MÓDULO 4: SEGMENTACIÓN ESPACIAL Y ASIGNACIÓN DE GRADILLA
# -------------------------------------------------------------------------
with tab4:
    st.header("🎯 Módulo 4: Detección Automática de ROIs y Mapeo")
    
    if "archivos_subidos" not in st.session_state or not st.session_state.archivos_subidos:
        st.warning("⚠️ Primero debés cargar imágenes en el Módulo 3.")
    else:
        st.write("Acá se ejecutará el algoritmo de segmentación S-G (oculto por defecto para el usuario).")
        st.info("Parámetros S-G espaciales preseteados internamente en modo óptimo.")

# -------------------------------------------------------------------------
# MÓDULO 5: ANÁLISIS CINÉTICO, CORTE DE SEDIMENTACIÓN Y DERIVADAS
# -------------------------------------------------------------------------
with tab5:
    st.header("📈 Módulo 5: Análisis Cinético y Tasas de Reacción")
    
    t_corte = st.slider("⏳ Tiempo de inicio de análisis (Ignorar sedimentación inicial en min):", 0.0, 10.0, 5.0, step=0.5)
    st.caption(f"Las derivadas dG/dt y dRatio/dt se calcularán únicamente para t > {t_corte} min.")

# -------------------------------------------------------------------------
# MÓDULO 6: EXPORTACIÓN DE DATOS, REPORTE PDF Y GOOGLE SHEETS
# -------------------------------------------------------------------------
with tab6:
    st.header("📄 Módulo 6: Descarga de Resultados y Registro Centralizado")
    st.write("Generación automática del PDF de reporte y sincronización de datos.")
