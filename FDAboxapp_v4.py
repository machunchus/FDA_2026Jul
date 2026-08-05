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

if "exp_id" not in st.session_state:
    st.session_state.exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Inicializar DataFrames vacíos para evitar KeyError absoluto
if "df_raw" not in st.session_state:
    st.session_state.df_raw = pd.DataFrame()

if "df_export" not in st.session_state:
    st.session_state.df_export = pd.DataFrame()

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
    modo_trabajo = st.radio("Acción a realizar:", ["➕ Crear Nuevo Ensayo", "🔍 Consultar Ensayo Existente (Histórico)"], horizontal=True)
    
    if modo_trabajo == "🔍 Consultar Ensayo Existente (Histórico)":
        st.info("ℹ️ Módulo histórico listo para vincular con la base de registros.")
        st.selectbox("Seleccione un Ensayo Previo:", ["EXP-20260805-185212 (Ensayo_Demo)"])
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
            if inicio_reaccion == "Tierra":
                tiempo_preinc = 0
                st.info("ℹ️ Al largar adicionando Tierra, la pre-incubación se fija en 0 min por defecto.")
            else:
                tiempo_preinc = st.number_input("Tiempo de pre-incubación (minutos):", min_value=1, value=15) if preincubacion else 0
            
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
        st.success("✅ Metadatos guardados.")

# -------------------------------------------------------------------------
# MÓDULO 2: PREPROCESAMIENTO DE MUESTRAS DE SUELO
# -------------------------------------------------------------------------
with tab2:
    st.header("🌱 Módulo 2: Preprocesamiento de Suelo")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tamizado = st.checkbox("¿Suelo Tamizado?", value=True)
        malla_tamiz = st.text_input("Apertura de Malla (ej: < 2mm):", value="< 2 mm") if tamizado else "N/A"
        
        opciones_secado = ["Secado al Aire", "Fresco / Humedad de Campo", "Secado en Estufa (30-40°C)", "Otro"]
        secado_sel = st.selectbox("Estado de Humedad / Secado:", opciones_secado)
        secado_detalle = st.text_input("Especifique estado de humedad/secado:") if secado_sel == "Otro" else secado_sel

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
# MÓDULO 3: CARGA DE FOTOS LOCALES Y METADATOS DE CÁMARA
# -------------------------------------------------------------------------
with tab3:
    st.header("📸 Módulo 3: Carga de Fotos y Parámetros de Cámara")
    
    with st.expander("📷 Registro de Parámetros de Cámara / Smartphone", expanded=False):
        c_cam1, c_cam2 = st.columns(2)
        with c_cam1:
            mod_sel = st.selectbox("Modelo de Smartphone / Cámara:", ["Samsung Galaxy S22", "Samsung Galaxy S21", "Xiaomi Redmi Note", "Motorola Edge", "Otro"])
            modelo_dispositivo = st.text_input("Especifique modelo:") if mod_sel == "Otro" else mod_sel
            iso_val = st.text_input("ISO / Sensibilidad:", value="100")
            ap_sel = st.selectbox("Apertura / Exposición:", ["1/10", "1/30", "1/60", "1/100", "Otro"])
            apertura_val = st.text_input("Especifique apertura:") if ap_sel == "Otro" else ap_sel
        with c_cam2:
            wb_sel = st.selectbox("Balance de Blancos (WB):", ["5500K (Soleado)", "Incandescente", "Fluorescente", "Automático", "Otro"])
            wb_val = st.text_input("Especifique WB:", value="5500K") if wb_sel == "Otro" else wb_sel
            enfoque_val = st.text_input("Modo de Enfoque:", value="Manual / Infinito")

    st.subheader("Subida de Imágenes desde Carpeta Local / Smartphone")
    archivos_subidos = st.file_uploader("Arrastrá la secuencia de fotos de la reacción (.jpg, .png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if archivos_subidos:
        st.session_state.archivos_subidos = sorted(archivos_subidos, key=lambda x: x.name)
        st.success(f"📁 {len(st.session_state.archivos_subidos)} imágenes cargadas correctamente.")
        st.session_state.tiene_t0 = st.checkbox("¿La primera foto corresponde a un Control / Blanco previo a la reacción (t=0)?")

# -------------------------------------------------------------------------
# MÓDULO 4: SEGMENTACIÓN AUTOMÁTICA DE ROIs Y ASIGNACIÓN DE MASAS
# -------------------------------------------------------------------------
with tab4:
    st.header("🎯 Módulo 4: Detección Automática de ROIs y Mapeo de Muestras")
    
    if "archivos_subidos" not in st.session_state or not st.session_state.archivos_subidos:
        st.warning("⚠️ Primero cargá las fotos en el Módulo 3.")
    else:
        st.markdown("### 🛠️ Ajuste y Visualización de Segmentación")
        
        # Parámetro ajustable por el usuario para forzar la detección si la imagen es rebelde
        prominencia = st.slider("Sensibilidad de detección (Bajar si detecta 0 posillos, subir si detecta de más):", min_value=1, max_value=50, value=10, step=1)
        
        bytes_primera = st.session_state.archivos_subidos[0].getvalue()
        img_np_0 = cv2.imdecode(np.frombuffer(bytes_primera, np.uint8), cv2.IMREAD_COLOR)
        
        if img_np_0 is not None:
            # Convertir a RGB para que Streamlit la muestre con los colores correctos
            img_rgb = cv2.cvtColor(img_np_0, cv2.COLOR_BGR2RGB)
            gray_0 = cv2.cvtColor(img_np_0, cv2.COLOR_BGR2GRAY)
            perfil_x = np.mean(gray_0, axis=0)
            
            p_sg_x = savgol_filter(perfil_x, window_length=51, polyorder=3)
            # Acá usamos la prominencia del slider
            picos_x, _ = find_peaks(p_sg_x, distance=img_np_0.shape[1]//8, prominence=prominencia)
            
            # --- SECCIÓN VISUAL ---
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("**1. Visión de la Cámara (Centro de los posillos detectados)**")
                img_display = img_rgb.copy()
                for px in picos_x:
                    # Dibujar línea vertical roja donde detecta el ROI
                    cv2.line(img_display, (px, 0), (px, img_display.shape[0]), (255, 0, 0), max(2, img_display.shape[1]//200)) 
                st.image(img_display, use_container_width=True)
                
            with col_v2:
                st.markdown("**2. Perfil de Intensidad (Radiografía del algoritmo)**")
                fig_p, ax_p = plt.subplots(figsize=(6, 4))
                ax_p.plot(p_sg_x, label="Perfil Lumínico", color='blue')
                ax_p.plot(picos_x, p_sg_x[picos_x], "x", color='red', markersize=10, label="Picos Encontrados")
                ax_p.set_xlabel("Ancho de la imagen (píxeles)")
                ax_p.set_ylabel("Intensidad Promedio")
                ax_p.legend()
                st.pyplot(fig_p)
            # ----------------------

            if len(picos_x) == 0:
                st.error("🚨 **Cero posillos detectados.** Probá bajar la Sensibilidad en la barra deslizante de arriba. Si sigue en 0, es muy probable que la foto esté torcida o muy oscura.")
            else:
                st.success(f"🎯 ROIs/Posillos detectados automáticamente: **{len(picos_x)} posillos**")
                
                st.subheader("Configuración de Muestras y Masa por Posillo (g)")
                cols_roi = st.columns(min(len(picos_x), 4)) if len(picos_x) > 0 else [st]
                
                datos_rois = {}
                for i, px in enumerate(picos_x):
                    col_idx = i % len(cols_roi)
                    with cols_roi[col_idx]:
                        st.markdown(f"**Posillo {i+1}** (x={px})")
                        nombre_roi = st.text_input(f"Nombre Muestra {i+1}", value=f"Muestra_{i+1}", key=f"n_roi_{i}")
                        masa_roi = st.number_input(f"Masa (g) Muestra {i+1}", min_value=0.01, value=1.00, step=0.05, key=f"m_roi_{i}")
                        datos_rois[f"ROI_{i+1}"] = {"x": px, "nombre": nombre_roi, "masa": masa_roi}
                
                st.session_state.datos_rois = datos_rois
                
                if st.button("🚀 Ejecutar Extracción de Intensidades en el Tiempo"):
                    progress_bar = st.progress(0)
                    resultados = []
                    
                    for idx, arch in enumerate(st.session_state.archivos_subidos):
                        file_bytes = arch.getvalue()
                        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            t_min = float(idx) # Incremento de 1 min por foto
                            
                            for roi_id, info in datos_rois.items():
                                px = info["x"]
                                y_center = img.shape[0] // 2
                                r_win = 15
                                patch = img[max(0, y_center-r_win):min(img.shape[0], y_center+r_win), max(0, px-r_win):min(img.shape[1], px+r_win)]
                                
                                b_val = np.mean(patch[:, :, 0])
                                g_val = np.mean(patch[:, :, 1])
                                r_val = np.mean(patch[:, :, 2])
                                
                                resultados.append({
                                    "Tiempo_min": t_min,
                                    "Imagen": arch.name,
                                    "ROI": roi_id,
                                    "Muestra": info["nombre"],
                                    "Masa_g": info["masa"],
                                    "R": r_val,
                                    "G": g_val,
                                    "B": b_val,
                                    "Ratio_GB": g_val / (b_val + 1e-6)
                                })
                        progress_bar.progress((idx + 1) / len(st.session_state.archivos_subidos))
                    
                    st.session_state.df_raw = pd.DataFrame(resultados)
                    st.success("✅ Extracción completada. Avanzá al Módulo 5.")

# -------------------------------------------------------------------------
# MÓDULO 5: ANÁLISIS CINÉTICO, CORTE DE SEDIMENTACIÓN Y DERIVADAS
# -------------------------------------------------------------------------
with tab5:
    st.header("📈 Módulo 5: Análisis Cinético y Tasas de Reacción")
    
    # Recuperamos el dataframe de forma segura
    df_raw = st.session_state.df_raw
    
    if df_raw.empty or "Tiempo_min" not in df_raw.columns:
        st.warning("⚠️ Todavía no hay datos extraídos. Volvé al Módulo 4 y hacé clic en '🚀 Ejecutar Extracción'.")
    else:
        st.subheader("⚙️ Configuración del Filtro Cinético")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            t_corte = st.slider("⏳ Ventana de Sedimentación (Ignorar primeros N min):", 0.0, 10.0, 5.0, step=0.5)
            st.caption(f"Los datos entre 0 y {t_corte} min serán ignorados para evitar artefactos de sedimentación.")
        
        with col_c2:
            normalizar_masa = st.checkbox("Normalizar Intensidades por Masa (Intensidad / g)", value=True)
            usar_ratio = st.checkbox("Analizar Ratio Verde/Azul (G/B) en lugar de Canal G puro", value=False)
            
        df_fit = df_raw[df_raw["Tiempo_min"] >= t_corte].copy()
        col_target = "Ratio_GB" if usar_ratio else "G"
        
        if normalizar_masa:
            df_fit["Valor_Analisis"] = df_fit[col_target] / df_fit["Masa_g"]
            label_y = f"{col_target} / g"
        else:
            df_fit["Valor_Analisis"] = df_fit[col_target]
            label_y = col_target

        st.subheader("📊 Gráficas Cinéticas y Derivadas (dG/dt)")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        df_export_list = []
        
        for muestra in df_fit["Muestra"].unique():
            sub_df = df_fit[df_fit["Muestra"] == muestra].sort_values("Tiempo_min")
            
            t_vals = sub_df["Tiempo_min"].values
            y_vals = sub_df["Valor_Analisis"].values
            
            ax1.plot(t_vals, y_vals, 'o-', label=muestra)
            
            if len(t_vals) >= 5:
                dt_prom = np.mean(np.diff(t_vals)) if len(t_vals) > 1 else 1.0
                window_l = 5 if len(t_vals) >= 5 else len(t_vals)
                if window_l % 2 == 0:
                    window_l -= 1
                    
                deriv_sg = savgol_filter(y_vals, window_length=window_l, polyorder=2, deriv=1, delta=dt_prom)
                ax2.plot(t_vals, deriv_sg, 's--', label=f"d({muestra})/dt")
                
                sub_df["Derivada_dY_dt"] = deriv_sg
            else:
                sub_df["Derivada_dY_dt"] = 0
                
            df_export_list.append(sub_df)

        ax1.set_xlabel("Tiempo (min)")
        ax1.set_ylabel(label_y)
        ax1.set_title(f"Cinética de Reacción (t > {t_corte} min)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()
        
        ax2.set_xlabel("Tiempo (min)")
        ax2.set_ylabel(f"d({label_y})/dt")
        ax2.set_title("Velocidad de Reacción (Derivada Temporal)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
        
        st.pyplot(fig)
        
        if df_export_list:
            st.session_state.df_export = pd.concat(df_export_list)
            st.success("✅ Análisis cinético completado. Avanzá al Módulo 6 para exportar los resultados.")

# -------------------------------------------------------------------------
# MÓDULO 6: EXPORTACIÓN DE DATOS Y REPORTE
# -------------------------------------------------------------------------
with tab6:
    st.header("📄 Módulo 6: Descarga de Resultados")
    
    if st.session_state.df_export.empty:
        st.warning("⚠️ No hay datos calculados todavía. Ejecutá el Módulo 5.")
    else:
        st.subheader("📥 Exportación de Tabla Completa (CSV)")
        csv_bytes = st.session_state.df_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="💾 Descargar CSV Completo (Cinética + Derivadas)",
            data=csv_bytes,
            file_name=f"{st.session_state.exp_id}_resultados.csv",
            mime="text/csv"
        )
        st.dataframe(st.session_state.df_export, use_container_width=True)
