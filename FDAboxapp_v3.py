import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import gc
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks

# =========================================================================
# CONFIGURACIÓN E INICIALIZACIÓN
# =========================================================================
st.set_page_config(page_title="Análisis FDA v3.3 - Plotly R/R0", layout="wide")
st.title("🔬 Análisis FDA - Matriz 3x4 Interactiva con Ratio R/R₀")

if "procesado" not in st.session_state:
    st.session_state.procesado = False

# =========================================================================
# CONTROLES DE PARÁMETROS
# =========================================================================
st.sidebar.header("⚙️ Configuración del Análisis")
opcion_rotar = st.sidebar.selectbox("Rotación de Cámara:", ["Sin Rotación", "180 Grados", "90 Grados Horario", "90 Grados Antihorario"])
dict_rotacion = {"Sin Rotación": None, "180 Grados": cv2.ROTATE_180, "90 Grados Horario": cv2.ROTATE_90_CLOCKWISE, "90 Grados Antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE}
rotacion_seleccionada = dict_rotacion[opcion_rotar]

metodo_estadistico = st.sidebar.radio("Cálculo de Intensidad ROI:", ["Mediana (Recomendado)", "Promedio"], index=0)

if "ancho_px" not in st.session_state: st.session_state.ancho_px = 1280
if "alto_px" not in st.session_state: st.session_state.alto_px = 960

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Filtros S-G: Detección Espacial")
prop_sg_y = st.sidebar.slider("Ventana S-G Y (%):", 0.5, 15.0, 1.5, step=0.1)
w_sg_y = max(5, int((prop_sg_y / 100.0) * st.session_state.alto_px))
if w_sg_y % 2 == 0: w_sg_y += 1

prop_sg_x = st.sidebar.slider("Ventana S-G X (%):", 0.1, 10.0, 0.5, step=0.1)
w_sg_x = max(5, int((prop_sg_x / 100.0) * st.session_state.ancho_px))
if w_sg_x % 2 == 0: w_sg_x += 1

poly_sg = st.sidebar.slider("Orden Polinomio Detección:", 2, 5, 4)
factor_reduccion = st.sidebar.slider("Factor reducción ROI:", 0.0, 0.45, 0.20, step=0.05)

# =========================================================================
# 1. CARGA DE IMÁGENES
# =========================================================================
st.subheader("🗂️ 1. Carga de Imágenes Secuenciales")
archivos_subidos = st.file_uploader("Arrastrá tus fotos aquí (La primera debe ser t=0)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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
    # 2. SEGMENTACIÓN DE ROIS (MATEMÁTICA V3)
    # =========================================================================
    st.markdown("---")
    st.subheader("📐 2. Diagnóstico Visual")
    
    ref_indices = list(range(0, num_img, freq_roi))
    st.session_state.rois_por_ref = {}
    n_rois_base = 0
    w_roi_fijo, h_roi_fijo = 30, 20
    
    try:
        for idx_panel, i in enumerate(ref_indices):
            archivo = archivos_ordenados[i]
            img_bytes = archivo.read()
            img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            archivo.seek(0)
            if rotacion_seleccionada is not None: img_bgr = cv2.rotate(img_bgr, rotacion_seleccionada)
            
            st.session_state.alto_px, st.session_state.ancho_px = img_bgr.shape[:2]
            canal_azul = img_bgr[:, :, 0]
            
            perfil_y = np.mean(canal_azul, axis=1)
            derivada_y = savgol_filter(perfil_y, window_length=w_sg_y, polyorder=poly_sg, deriv=1)
            y_min = int(np.argmax(derivada_y))
            y_max = int(np.argmin(derivada_y[y_min:]) + y_min)
            y_central = (y_min + y_max) // 2
            alto_banda = y_max - y_min

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
            else:
                if len(lista_centros_x) != n_rois_base:
                    lista_centros_x, y_central = st.session_state.rois_por_ref[0]
                st.session_state.rois_por_ref[i] = (lista_centros_x, y_central)

            del img_bgr, canal_azul, franja_azul; gc.collect()
            
        st.session_state.w_roi_fijo = w_roi_fijo
        st.session_state.h_roi_fijo = h_roi_fijo
        st.session_state.n_rois_base = n_rois_base
        st.success(f"Detección correcta: {n_rois_base} ROIs encontrados.")

    except Exception as e:
        st.error(f"Error en segmentación: {e}"); st.stop()

    # =========================================================================
    # 3. METADATOS (NOMBRES Y MASAS)
    # =========================================================================
    st.markdown("---")
    st.subheader("🏷️ 3. Identificación de Muestras y Masa Pesada (g)")
    
    nombres_muestras, masas_muestras = [], []
    with st.expander("📝 Formulario de Muestras", expanded=True):
        for i in range(st.session_state.n_rois_base):
            c_id, c_name, c_mass = st.columns([1, 2, 2])
            c_id.write(f"**ROI {i+1}**")
            k_name, k_mass = f"roi_name_val_{i}", f"roi_mass_val_{i}"
            if k_name not in st.session_state: st.session_state[k_name] = f"Muestra_{i+1}"
            if k_mass not in st.session_state: st.session_state[k_mass] = 1.0000
            val_name = c_name.text_input(f"Label {i+1}", value=st.session_state[k_name], label_visibility="collapsed", key=f"ui_str_{i}")
            val_mass = c_mass.number_input(f"Mass {i+1}", value=st.session_state[k_mass], min_value=0.0001, step=0.0001, format="%.4f", label_visibility="collapsed", key=f"ui_num_{i}")
            st.session_state[k_name], st.session_state[k_mass] = val_name, val_mass
            nombres_muestras.append(val_name); masas_muestras.append(val_mass)

    # =========================================================================
    # 4. EXTRACCIÓN DE SEÑALES
    # =========================================================================
    st.markdown("---")
    if st.button("▶️ Lanzar Procesamiento de Lote"):
        if st.session_state.n_rois_base > 0:
            barra = st.progress(0)
            tiempos_dt = [datetime.strptime(f.name.rsplit('.', 1)[0], "%Y-%m-%d_%H-%M-%S") for f in archivos_ordenados]
            t0 = min(tiempos_dt)
            t_rel_min = np.array([(t - t0).total_seconds() / 60.0 for t in tiempos_dt])

            n_r = st.session_state.n_rois_base
            h_verde, h_azul = np.zeros((num_img, n_r)), np.zeros((num_img, n_r))
            func_est = np.median if "Mediana" in metodo_estadistico else np.mean

            for idx, archivo in enumerate(archivos_ordenados):
                barra.progress(int((idx + 1) / num_img * 100))
                ref_idx = (idx // freq_roi) * freq_roi
                centros_tuplas, y_cent_actual = st.session_state.rois_por_ref[ref_idx]
                centros_x_act = [b[2] for b in centros_tuplas]
                
                frame_bgr = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), cv2.IMREAD_COLOR)
                if rotacion_seleccionada is not None: frame_bgr = cv2.rotate(frame_bgr, rotacion_seleccionada)
                
                c_a, c_v = frame_bgr[:, :, 0], frame_bgr[:, :, 1]
                w_r, h_r = st.session_state.w_roi_fijo, st.session_state.h_roi_fijo
                
                for r_idx, cx in enumerate(centros_x_act):
                    x1, x2 = cx - w_r // 2, cx + w_r // 2
                    y1, y2 = y_cent_actual - h_r // 2, y_cent_actual + h_r // 2
                    h_verde[idx, r_idx] = func_est(c_v[y1:y2, x1:x2])
                    h_azul[idx, r_idx] = func_est(c_a[y1:y2, x1:x2])
                del frame_bgr; gc.collect()

            st.session_state.archivos_nombres = [f.name for f in archivos_ordenados]
            st.session_state.t_rel_min = t_rel_min
            st.session_state.h_verde, st.session_state.h_azul = h_verde, h_azul
            st.session_state.g0, st.session_state.a0 = h_verde[0, :], h_azul[0, :]
            st.session_state.nombres_finales, st.session_state.masas_finales = nombres_muestras, masas_muestras
            st.session_state.num_img = num_img
            st.session_state.procesado = True

    # =========================================================================
    # 5. MATRIZ INTERACTIVA (3 COLUMNAS x 4 FILAS)
    # =========================================================================
    if st.session_state.procesado:
        t = st.session_state.t_rel_min
        lbls = st.session_state.nombres_finales
        masas = np.array(st.session_state.masas_finales)
        num_img = st.session_state.num_img
        n_r = st.session_state.n_rois_base

        st.markdown("---")
        st.subheader("⏳ Control de Cutoff de Sedimentación")
        t_cutoff = st.slider("Tiempo de corte Cutoff (minutos):", 0.0, float(np.max(t)), 2.0, step=0.5)

        # ---------------------------------------------------------------------
        # MATEMÁTICA ANALÍTICA DE CANALES Y RATIOS (R/R0)
        # ---------------------------------------------------------------------
        # Fila 1: Valores Crudos
        g_crudo = st.session_state.h_verde
        a_crudo = st.session_state.h_azul
        r_crudo = np.where(a_crudo == 0, 1e-6, g_crudo / a_crudo)

        # Basales t=0
        g0 = st.session_state.g0
        a0 = st.session_state.a0
        r0 = np.where(a0 == 0, 1e-6, g0 / a0)

        # Fila 2: Netos / Normalizados
        g_norm = g_crudo - g0
        a_norm = a_crudo - a0
        r_norm = r_crudo / r0  # R / R0

        # Filtro Temporal S-G por Cutoff
        mask_cutoff = t >= t_cutoff
        t_filt = t[mask_cutoff]
        dt_prom = np.mean(np.diff(t)) if len(t) > 1 else 1.0

        w_cin = w_sg_cin if w_sg_cin <= num_img else (num_img if num_img % 2 != 0 else num_img - 1)
        poly_cin = poly_sg_cin if poly_sg_cin < w_cin else w_cin - 1

        # Fila 3: Velocidades (> Cutoff)
        v_gnorm = np.full_like(g_norm, np.nan)
        v_anorm = np.full_like(a_norm, np.nan)
        v_rnorm = np.full_like(r_norm, np.nan)

        if len(t_filt) >= w_cin:
            for r in range(n_r):
                dg = savgol_filter(g_norm[mask_cutoff, r], w_cin, poly_cin, deriv=1, delta=dt_prom)
                da = savgol_filter(a_norm[mask_cutoff, r], w_cin, poly_cin, deriv=1, delta=dt_prom)
                dr = savgol_filter(r_norm[mask_cutoff, r], w_cin, poly_cin, deriv=1, delta=dt_prom)
                m = w_cin // 2
                dg[:m], dg[-m:] = np.nan, np.nan
                da[:m], da[-m:] = np.nan, np.nan
                dr[:m], dr[-m:] = np.nan, np.nan
                v_gnorm[mask_cutoff, r] = dg
                v_anorm[mask_cutoff, r] = da
                v_rnorm[mask_cutoff, r] = dr

        # Fila 4: Velocidades / Masa
        v_gnorm_m = v_gnorm / masas
        v_anorm_m = v_anorm / masas
        v_rnorm_m = v_rnorm / masas

        # Generador dinámico de figuras Plotly
        def crear_figura_plotly(x_data, y_matrix, titulo, y_label, cutoff_val=None):
            fig = go.Figure()
            for r in range(n_r):
                fig.add_trace(go.Scatter(x=x_data, y=y_matrix[:, r], mode='lines+markers', name=lbls[r]))
            if cutoff_val is not None:
                fig.add_vline(x=cutoff_val, line_dash="dash", line_color="red", annotation_text=f"Cutoff {cutoff_val}m")
            fig.update_layout(
                title=dict(text=titulo, font=dict(size=12)),
                xaxis_title="Tiempo (min)", yaxis_title=y_label,
                margin=dict(l=20, r=20, t=35, b=20), height=320,
                legend=dict(font=dict(size=9), orientation="h", y=-0.25)
            )
            return fig

        st.markdown("---")
        st.subheader("📊 Panel de Gráficas: Verde | Azul | Ratio (R/R₀)")

        # FILA 1: CRUDOS
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(crear_figura_plotly(t, g_crudo, "1A) Verde Crudo (G)", "Intensidad", t_cutoff), use_container_width=True)
        with c2: st.plotly_chart(crear_figura_plotly(t, a_crudo, "1B) Azul Crudo (A)", "Intensidad", t_cutoff), use_container_width=True)
        with c3: st.plotly_chart(crear_figura_plotly(t, r_crudo, "1C) Ratio Crudo (G/A)", "Ratio (G/A)", t_cutoff), use_container_width=True)

        # FILA 2: NETOS / NORMALIZADOS
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(crear_figura_plotly(t, g_norm, "2A) Verde Neto (G - G0)", "Δ Intensidad"), use_container_width=True)
        with c2: st.plotly_chart(crear_figura_plotly(t, a_norm, "2B) Azul Neto (A - A0)", "Δ Intensidad"), use_container_width=True)
        with c3: st.plotly_chart(crear_figura_plotly(t, r_norm, "2C) Ratio Normalizado (R / R0)", "Ratio (R/R0)"), use_container_width=True)

        # FILA 3: VELOCIDADES
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(crear_figura_plotly(t, v_gnorm, "3A) Vel. Verde Neto", "d(G-G0)/dt"), use_container_width=True)
        with c2: st.plotly_chart(crear_figura_plotly(t, v_anorm, "3B) Vel. Azul Neto", "d(A-A0)/dt"), use_container_width=True)
        with c3: st.plotly_chart(crear_figura_plotly(t, v_rnorm, "3C) Vel. Ratio (R/R0)", "d(R/R0)/dt"), use_container_width=True)

        # FILA 4: VELOCIDADES / MASA
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(crear_figura_plotly(t, v_gnorm_m, "4A) Vel. Verde / Masa", "Unidades / (min·g)"), use_container_width=True)
        with c2: st.plotly_chart(crear_figura_plotly(t, v_anorm_m, "4B) Vel. Azul / Masa", "Unidades / (min·g)"), use_container_width=True)
        with c3: st.plotly_chart(crear_figura_plotly(t, v_rnorm_m, "4C) Vel. Ratio (R/R0) / Masa", "Unidades / (min·g)"), use_container_width=True)

        # =========================================================================
        # CONSTRUCCIÓN Y DESCARGA CSV
        # =========================================================================
        st.markdown("---")
        st.subheader("💾 Exportación de Resultados")
        cols = ["Archivo", "Tiempo_Min"]
        for r in range(n_r):
            cols.extend([
                f"{lbls[r]}_G_crudo", f"{lbls[r]}_A_crudo", f"{lbls[r]}_R_crudo",
                f"{lbls[r]}_G-G0", f"{lbls[r]}_A-A0", f"{lbls[r]}_R/R0",
                f"{lbls[r]}_vG", f"{lbls[r]}_vA", f"{lbls[r]}_v(R/R0)",
                f"{lbls[r]}_vG_m", f"{lbls[r]}_vA_m", f"{lbls[r]}_v(R/R0)_m",
                f"{lbls[r]}_Masa_g"
            ])
            
        datos = []
        for i_img in range(num_img):
            f = [st.session_state.archivos_nombres[i_img], t[i_img]]
            for r in range(n_r):
                f.extend([
                    g_crudo[i_img, r], a_crudo[i_img, r], r_crudo[i_img, r],
                    g_norm[i_img, r], a_norm[i_img, r], r_norm[i_img, r],
                    v_gnorm[i_img, r], v_anorm[i_img, r], v_rnorm[i_img, r],
                    v_gnorm_m[i_img, r], v_anorm_m[i_img, r], v_rnorm_m[i_img, r],
                    masas[r]
                ])
            datos.append(f)

        df_exp = pd.DataFrame(datos, columns=cols)
        st.download_button("📥 Descargar Tabla Completa (CSV)", df_exp.to_csv(index=False).encode('utf-8'), f"fda_matrix_r_r0_{int(time.time())}.csv", "text/csv")
        st.dataframe(df_exp.head(10).style.format(precision=4, na_rep='NaN'), use_container_width=True)
