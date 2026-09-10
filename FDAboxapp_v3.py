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
# CONFIGURACIÓN E INICIALIZACIÓN DEL ESTADO
# =========================================================================
st.set_page_config(page_title="Análisis FDA v3.1 - Con Normalización y Cutoff", layout="wide")
st.title("🔬 Herramienta de Análisis FDA (Sensor de Suelo) - v3.1")

if "procesado" not in st.session_state:
    st.session_state.procesado = False
    st.session_state.df = None

# =========================================================================
# BARRA LATERAL (CONTROLES DE PARÁMETROS)
# =========================================================================
st.sidebar.header("⚙️ Configuración del Análisis")
opcion_rotar = st.sidebar.selectbox("Rotación de Cámara:", ["Sin Rotación", "180 Grados", "90 Grados Horario", "90 Grados Antihorario"])
dict_rotacion = {"Sin Rotación": None, "180 Grados": cv2.ROTATE_180, "90 Grados Horario": cv2.ROTATE_90_CLOCKWISE, "90 Grados Antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE}
rotacion_seleccionada = dict_rotacion[opcion_rotar]

metodo_estadistico = st.sidebar.radio("Cálculo de Intensidad de ROI:", ["Mediana (Recomendado)", "Promedio"], index=0)

if "ancho_px" not in st.session_state: st.session_state.ancho_px = 1280
if "alto_px" not in st.session_state: st.session_state.alto_px = 960

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Filtros S-G: Detección Espacial")
prop_sg_y = st.sidebar.slider("Ventana S-G en Y (%):", 0.5, 15.0, 1.5, step=0.1)
w_sg_y = max(5, int((prop_sg_y / 100.0) * st.session_state.alto_px))
if w_sg_y % 2 == 0: w_sg_y += 1

prop_sg_x = st.sidebar.slider("Ventana S-G en X (%):", 0.1, 10.0, 0.5, step=0.1)
w_sg_x = max(5, int((prop_sg_x / 100.0) * st.session_state.ancho_px))
if w_sg_x % 2 == 0: w_sg_x += 1

poly_sg = st.sidebar.slider("Orden Polinomio Detección:", 2, 5, 4)
factor_reduccion = st.sidebar.slider("Factor reducción ROI:", 0.0, 0.45, 0.20, step=0.05)

# =========================================================================
# 1. CARGA DE IMÁGENES SECUENCIALES
# =========================================================================
st.subheader("🗂️ 1. Carga de Imágenes Secuenciales")
archivos_subidos = st.file_uploader("Arrastrá tus fotos aquí (Asegurate que la 1° sea t=0 / Sedimentado)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if archivos_subidos:
    archivos_ordenados = sorted(archivos_subidos, key=lambda x: x.name)
    num_img = len(archivos_ordenados)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Rastreo Dinámico de ROIs")
    freq_roi = st.sidebar.slider("Frec. re-cálculo ROI (cada N fotos):", 1, num_img, num_img)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Filtros S-G: Cinéticas Temporales")
    w_sg_cin = st.sidebar.slider(f"Ventana S-G Temporal (5 a {num_img}):", min_value=5, max_value=max(5, num_img), value=min(31, max(5, num_img)), step=2)
    poly_sg_cin = st.sidebar.slider("Orden Polinomio Derivada:", 1, 5, 2)
    
    # =========================================================================
    # 2. DIAGNÓSTICO VISUAL Y RASTREO DINÁMICO
    # =========================================================================
    st.markdown("---")
    st.subheader("📐 2. Diagnóstico Visual y Rastreo Dinámico")
    
    ref_indices = list(range(0, num_img, freq_roi))
    st.session_state.rois_por_ref = {}
    n_rois_base = 0
    w_roi_fijo, h_roi_fijo = 30, 20
    
    st.write(f"Se re-calcularán las coordenadas ROI en {len(ref_indices)} imágenes de referencia.")
    galeria_cols = st.columns(4)
    
    try:
        for idx_panel, i in enumerate(ref_indices):
            archivo = archivos_ordenados[i]
            img_bytes = archivo.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            archivo.seek(0)
            if rotacion_seleccionada is not None: img_bgr = cv2.rotate(img_bgr, rotacion_seleccionada)
            
            st.session_state.alto_px, st.session_state.ancho_px = img_bgr.shape[:2]
            canal_azul = img_bgr[:, :, 0]
            
            # Derivadas Y
            perfil_y = np.mean(canal_azul, axis=1)
            derivada_y = savgol_filter(perfil_y, window_length=w_sg_y, polyorder=poly_sg, deriv=1)
            y_min = int(np.argmax(derivada_y))
            y_max = int(np.argmin(derivada_y[y_min:]) + y_min)
            y_central = (y_min + y_max) // 2
            alto_banda = y_max - y_min

            # Derivadas X
            franja_azul = canal_azul[y_min:y_max, :]
            perfil_x = np.mean(franja_azul, axis=0)
            derivada_x = savgol_filter(perfil_x, window_length=w_sg_x, polyorder=poly_sg, deriv=1)
            umbral_x = np.max(np.abs(derivada_x)) * 0.15
            bordes_izq, _ = find_peaks(derivada_x, height=umbral_x, distance=max(15, st.session_state.ancho_px // 90))
            bordes_der, _ = find_peaks(-derivada_x, height=umbral_x, distance=max(15, st.session_state.ancho_px // 90))

            lista_centros_x = []
            for b_izq in bordes_izq:
                b_der_cands = bordes_der[bordes_der > b_izq]
                if len(b_der_cands) > 0:
                    b_der = b_der_cands[0]
                    if int(st.session_state.ancho_px * 0.008) < (b_der - b_izq) < int(st.session_state.ancho_px * 0.08):
                        lista_centros_x.append((b_izq, b_der, (b_izq + b_der) // 2))

            lista_centros_x.sort(key=lambda x: x[0])
            
            if i == 0:
                n_rois_base = len(lista_centros_x)
                if n_rois_base > 0:
                    ancho_min = min([b[1] - b[0] for b in lista_centros_x])
                    red_px = int(ancho_min * factor_reduccion * 2)
                    w_roi_fijo, h_roi_fijo = ancho_min - red_px, alto_banda - red_px
                
                st.session_state.rois_por_ref[i] = (lista_centros_x, y_central)
                
                st.markdown("**Gráficas de Diagnóstico (Señal cruda vs S-G) - Basadas en la primera foto:**")
                col_diag1, col_diag2 = st.columns(2)
                
                with col_diag1:
                    fig_y, ax1_y = plt.subplots(figsize=(5, 3))
                    ax1_y.plot(perfil_y, color='green', label='Intensidad')
                    ax1_y.set_xlabel("Posición Pixel Y"); ax1_y.set_ylabel("Intensidad", color='green')
                    
                    ax2_y = ax1_y.twinx()
                    ax2_y.plot(derivada_y, color='red', alpha=0.7, label='1ª Derivada S-G')
                    ax2_y.set_ylabel("Gradiente (S-G)", color='red')
                    
                    ax1_y.axvline(y_min, color='blue', linestyle='--', label=f'Borde ({y_min})')
                    ax1_y.axvline(y_max, color='blue', linestyle='--')
                    ax1_y.axvline(y_central, color='orange', linestyle='-', label=f'Centro ({y_central})')
                    
                    fig_y.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85), fontsize=7)
                    plt.title("Segmentación Eje Y")
                    st.pyplot(fig_y); plt.close(fig_y)
                    
                with col_diag2:
                    fig_x, ax1_x = plt.subplots(figsize=(5, 3))
                    ax1_x.plot(perfil_x, color='green', label='Intensidad')
                    ax1_x.set_xlabel("Posición Pixel X"); ax1_x.set_ylabel("Intensidad", color='green')
                    
                    ax2_x = ax1_x.twinx()
                    ax2_x.plot(derivada_x, color='red', alpha=0.7, label='1ª Derivada S-G')
                    ax2_x.set_ylabel("Gradiente (S-G)", color='red')
                    
                    if len(lista_centros_x) > 0:
                        ax1_x.axvline(lista_centros_x[0][0], color='blue', linestyle='--', label='Bordes')
                        ax1_x.axvline(lista_centros_x[0][1], color='blue', linestyle='--')
                        for c in lista_centros_x: ax1_x.axvline(c[2], color='orange', linestyle='-', alpha=0.5)
                        
                    fig_x.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85), fontsize=7)
                    plt.title("Detección de Flancos Eje X")
                    st.pyplot(fig_x); plt.close(fig_x)
            else:
                if len(lista_centros_x) != n_rois_base:
                    lista_centros_x, y_central = st.session_state.rois_por_ref[0]
                st.session_state.rois_por_ref[i] = (lista_centros_x, y_central)

            img_mascara = img_bgr.copy()
            for r_idx, (b_izq, b_der, cx) in enumerate(st.session_state.rois_por_ref[i][0]):
                y_c = st.session_state.rois_por_ref[i][1]
                cv2.rectangle(img_mascara, (b_izq, y_c - alto_banda//2), (b_der, y_c + alto_banda//2), (0, 255, 0), 2)
                x1_roi, x2_roi = cx - w_roi_fijo // 2, cx + w_roi_fijo // 2
                y1_roi, y2_roi = y_c - h_roi_fijo // 2, y_c + h_roi_fijo // 2
                cv2.rectangle(img_mascara, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 215, 255), 1)
                cv2.putText(img_mascara, f"R{r_idx+1}", (b_izq, y_c - alto_banda//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            img_resized = cv2.resize(img_mascara, (0,0), fx=0.3, fy=0.3)
            with galeria_cols[idx_panel % 4]:
                st.image(img_resized, channels="BGR", caption=f"Ref: {archivo.name}")
                
            del img_bgr, canal_azul, franja_azul, img_mascara, img_resized; gc.collect()
            
        st.session_state.w_roi_fijo = w_roi_fijo
        st.session_state.h_roi_fijo = h_roi_fijo
        st.session_state.n_rois_base = n_rois_base

    except Exception as e:
        st.error(f"Error en rastreo de ROIs: {e}"); st.stop()

    # =========================================================================
    # 3. MENÚ DE MUESTRAS
    # =========================================================================
    st.markdown("---")
    st.subheader("🏷️ 3. Identificación de Muestras y Metadatos Analíticos")
    
    nombres_muestras = []
    masas_muestras = []

    with st.expander("📝 Formulario de Carga: Nombres y Masas", expanded=True):
        c_h1, c_h2, c_h3 = st.columns([1, 2, 2])
        c_h1.markdown("**Código Base**")
        c_h2.markdown("**Nombre Asignado de la Muestra**")
        c_h3.markdown("**Masa Pesada (g)**")
        
        for i in range(st.session_state.n_rois_base):
            c_id, c_name, c_mass = st.columns([1, 2, 2])
            c_id.write(f"**ROI {i+1}**")
            
            key_name, key_mass = f"roi_name_val_{i}", f"roi_mass_val_{i}"
            if key_name not in st.session_state: st.session_state[key_name] = f"Muestra_{i+1}"
            if key_mass not in st.session_state: st.session_state[key_mass] = 1.0000
                
            val_name = c_name.text_input(f"Label {i+1}", value=st.session_state[key_name], label_visibility="collapsed", key=f"ui_str_{i}")
            val_mass = c_mass.number_input(f"Mass {i+1}", value=st.session_state[key_mass], min_value=0.0001, step=0.0001, format="%.4f", label_visibility="collapsed", key=f"ui_num_{i}")
            
            st.session_state[key_name] = val_name; st.session_state[key_mass] = val_mass
            nombres_muestras.append(val_name); masas_muestras.append(val_mass)

    # =========================================================================
    # 4. PROCESAMIENTO KINÉTICO POR LOTE CON G0 Y A0
    # =========================================================================
    st.markdown("---")
    st.subheader("🚀 4. Ejecución del Perfil Cinético Completo")
    
    if st.button("▶️ Lanzar Procesamiento de Lote"):
        if st.session_state.n_rois_base > 0:
            barra_progreso = st.progress(0)
            
            tiempos_dt = []
            for f in archivos_ordenados:
                try: tiempos_dt.append(datetime.strptime(f.name.rsplit('.', 1)[0], "%Y-%m-%d_%H-%M-%S"))
                except ValueError: st.error(f"Falla de parseo: {f.name} no cumple el formato AAAA-MM-DD_HH-mm-ss"); st.stop()
                    
            t0 = min(tiempos_dt)
            t_rel_min = np.array([(t - t0).total_seconds() / 60.0 for t in tiempos_dt])

            n_r = st.session_state.n_rois_base
            h_verde, h_azul = np.zeros((num_img, n_r)), np.zeros((num_img, n_r))

            func_est = np.median if "Mediana" in metodo_estadistico else np.mean

            for idx, archivo in enumerate(archivos_ordenados):
                barra_progreso.progress(int((idx + 1) / num_img * 100))
                
                ref_idx = (idx // freq_roi) * freq_roi
                centros_tuplas, y_cent_actual = st.session_state.rois_por_ref[ref_idx]
                centros_x_act = [b[2] for b in centros_tuplas]
                
                img_bytes = archivo.read()
                frame_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if rotacion_seleccionada is not None: frame_bgr = cv2.rotate(frame_bgr, rotacion_seleccionada)
                
                c_a, c_v = frame_bgr[:, :, 0], frame_bgr[:, :, 1]
                w_r, h_r = st.session_state.w_roi_fijo, st.session_state.h_roi_fijo
                
                for r_idx, cx in enumerate(centros_x_act):
                    x1, x2 = cx - w_r // 2, cx + w_r // 2
                    y1, y2 = y_cent_actual - h_r // 2, y_cent_actual + h_r // 2
                    m_v, m_a = func_est(c_v[y1:y2, x1:x2]), func_est(c_a[y1:y2, x1:x2])
                    h_verde[idx, r_idx] = m_v; h_azul[idx, r_idx] = m_a
                
                del frame_bgr, c_a, c_v
                if idx % 10 == 0: gc.collect()

            # Extraemos los basales G0 y A0 (Primera foto, t=0)
            g0 = h_verde[0, :]
            a0 = h_azul[0, :]

            st.session_state.archivos_nombres = [f.name for f in archivos_ordenados]
            st.session_state.t_rel_min = t_rel_min
            st.session_state.h_verde, st.session_state.h_azul = h_verde, h_azul
            st.session_state.g0, st.session_state.a0 = g0, a0
            st.session_state.nombres_finales, st.session_state.masas_finales = nombres_muestras, masas_muestras
            st.session_state.num_img = num_img
            st.session_state.procesado = True

    # =========================================================================
    # 5. ANÁLISIS DE CURVAS Y DERIVADAS CON CUTOFF
    # =========================================================================
    if st.session_state.procesado:
        t = st.session_state.t_rel_min
        lbls = st.session_state.nombres_finales
        masas = np.array(st.session_state.masas_finales)
        num_img = st.session_state.num_img
        n_r = st.session_state.n_rois_base
        
        st.markdown("---")
        st.subheader("⏳ Ventana de Corte de Sedimentación (Cutoff)")
        
        t_max_val = float(np.max(t)) if len(t) > 1 else 10.0
        t_cutoff = st.slider("Minutos a ignorar por sedimentación inicial (Cutoff):", 0.0, float(np.percentile(t, 50)), 2.0, step=0.5)

        # Cálculos de Matrices Normalizadas
        # G_norm = G - G0 | A_norm = A - A0
        g_norm = st.session_state.h_verde - st.session_state.g0
        a_norm = st.session_state.h_azul - st.session_state.a0
        
        # Evitar división por cero en el Ratio Normalizado
        a_norm_safe = np.where(a_norm == 0, 1e-6, a_norm)
        ratio_norm = g_norm / a_norm_safe

        # Máscara temporal para filtrado de Cutoff
        mask_cutoff = t >= t_cutoff
        t_filtrado = t[mask_cutoff]
        dt_promedio = np.mean(np.diff(t)) if len(t) > 1 else 1.0

        w_cin = w_sg_cin if w_sg_cin <= num_img else (num_img if num_img % 2 != 0 else num_img - 1)
        poly_cin = poly_sg_cin if poly_sg_cin < w_cin else w_cin - 1

        # Inicializar matrices de derivadas
        d_gnorm_dt = np.full_like(g_norm, np.nan)
        d_anorm_dt = np.full_like(a_norm, np.nan)

        if len(t_filtrado) >= w_cin:
            for r in range(n_r):
                # Derivada de G_norm filtrada por Cutoff
                dg_val = savgol_filter(g_norm[mask_cutoff, r], window_length=w_cin, polyorder=poly_cin, deriv=1, delta=dt_promedio)
                da_val = savgol_filter(a_norm[mask_cutoff, r], window_length=w_cin, polyorder=poly_cin, deriv=1, delta=dt_promedio)
                
                # Anular los bordes del filtro S-G
                margen = w_cin // 2
                dg_val[:margen], dg_val[-margen:] = np.nan, np.nan
                da_val[:margen], da_val[-margen:] = np.nan, np.nan
                
                d_gnorm_dt[mask_cutoff, r] = dg_val
                d_anorm_dt[mask_cutoff, r] = da_val
        else:
            st.warning(f"⚠️ Cantidad de puntos por encima del Cutoff ({len(t_filtrado)}) es inferior a la ventana S-G ({w_cin}). Reducí el Cutoff o la ventana S-G.")

        # Normalizaciones por masa
        d_gnorm_dt_masa = d_gnorm_dt / masas
        d_anorm_dt_masa = d_anorm_dt / masas

        # =========================================================================
        # DESPLEGABLE DE GRÁFICAS A - I
        # =========================================================================
        st.markdown("---")
        st.subheader("📊 Gráficas de Intensidades y Cinéticas (A - I)")
        
        # FILA 1: A, B, C
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, st.session_state.h_verde[:, r], label=lbls[r])
            ax.axvline(t_cutoff, color='red', linestyle='--', label=f'Cutoff ({t_cutoff} min)')
            ax.axvspan(0, t_cutoff, color='red', alpha=0.1)
            ax.set_title("A) Intensidad Verde Crudo vs t"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        with col_b:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, g_norm[:, r], label=lbls[r])
            ax.set_title("B) Verde Normalizado (G - G0) vs t"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        with col_c:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, st.session_state.h_azul[:, r], label=lbls[r])
            ax.set_title("C) Intensidad Azul Crudo vs t"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        # FILA 2: D, E, F
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, a_norm[:, r], label=lbls[r])
            ax.set_title("D) Azul Normalizado (A - A0) vs t"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        with col_e:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, ratio_norm[:, r], label=lbls[r])
            ax.set_title("E) Ratio (G - G0)/(A - A0) vs t"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        with col_f:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, d_gnorm_dt[:, r], label=lbls[r])
            ax.set_title("F) Vel. d(G-G0)/dt vs t (>Cutoff)"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        # FILA 3: G, H, I
        col_g, col_h, col_i = st.columns(3)
        with col_g:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, d_anorm_dt[:, r], label=lbls[r])
            ax.set_title("G) Vel. d(A-A0)/dt vs t (>Cutoff)"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        with col_h:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, d_gnorm_dt_masa[:, r], label=lbls[r])
            ax.set_title("H) Vel. Verde / Masa Suelo vs t"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        with col_i:
            fig, ax = plt.subplots(figsize=(5,3.8))
            for r in range(n_r): ax.plot(t, d_anorm_dt_masa[:, r], label=lbls[r])
            ax.set_title("I) Vel. Azul / Masa Suelo vs t"); ax.set_xlabel("Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
            st.pyplot(fig); plt.close(fig)

        # =========================================================================
        # CONSTRUCCIÓN DE DATAFRAME DE EXPORTACIÓN COMPLETO
        # =========================================================================
        cols = ["Nombre_Archivo", "Tiempo_Rel_Minutos"]
        for r in range(n_r):
            cols.extend([
                f"{lbls[r]}_G_crudo", f"{lbls[r]}_G0", f"{lbls[r]}_G-G0",
                f"{lbls[r]}_A_crudo", f"{lbls[r]}_A0", f"{lbls[r]}_A-A0",
                f"{lbls[r]}_Ratio_Norm",
                f"{lbls[r]}_d(G-G0)/dt", f"{lbls[r]}_d(A-A0)/dt",
                f"{lbls[r]}_d(G-G0)/dt_por_g", f"{lbls[r]}_d(A-A0)/dt_por_g",
                f"{lbls[r]}_Masa_g"
            ])
            
        datos = []
        for i_img in range(num_img):
            fila = [st.session_state.archivos_nombres[i_img], t[i_img]]
            for r in range(n_r):
                fila.extend([
                    st.session_state.h_verde[i_img, r], st.session_state.g0[r], g_norm[i_img, r],
                    st.session_state.h_azul[i_img, r], st.session_state.a0[r], a_norm[i_img, r],
                    ratio_norm[i_img, r],
                    d_gnorm_dt[i_img, r], d_anorm_dt[i_img, r],
                    d_gnorm_dt_masa[i_img, r], d_anorm_dt_masa[i_img, r],
                    masas[r]
                ])
            datos.append(fila)

        df_export = pd.DataFrame(datos, columns=cols)

        st.markdown("---")
        st.subheader("💾 Descarga de Resultados")
        st.download_button(
            "📥 Descargar Tabla Completa con Normalizaciones y Derivadas (CSV)",
            df_export.to_csv(index=False).encode('utf-8'),
            f"cinetica_fda_v3.1_{int(time.time())}.csv",
            "text/csv"
        )
        st.dataframe(df_export.head(10).style.format(precision=4, na_rep='NaN'), use_container_width=True)
