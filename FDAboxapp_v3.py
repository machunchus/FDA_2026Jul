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
st.set_page_config(page_title="Análisis FDA - Interactividad Total", layout="wide")
st.title("🔬 Análisis FDA - Motor Dinámico v4")

if "procesado" not in st.session_state:
    st.session_state.procesado = False

# =========================================================================
# CONTROLES DE PARÁMETROS (BARRA LATERAL)
# =========================================================================
st.sidebar.header("⚙️ Configuración")
opcion_rotar = st.sidebar.selectbox("Rotación de Cámara:", ["Sin Rotación", "180 Grados", "90 Grados Horario", "90 Grados Antihorario"])
dict_rotacion = {"Sin Rotación": None, "180 Grados": cv2.ROTATE_180, "90 Grados Horario": cv2.ROTATE_90_CLOCKWISE, "90 Grados Antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE}
rotacion_seleccionada = dict_rotacion[opcion_rotar]

metodo_estadistico = st.sidebar.radio("Cálculo de Intensidad:", ["Mediana", "Promedio"], index=0)

if "ancho_px" not in st.session_state: st.session_state.ancho_px = 1280
if "alto_px" not in st.session_state: st.session_state.alto_px = 960

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Detección Espacial (ROIs)")
n_rois_esperado = st.sidebar.number_input("Número exacto de muestras (ROIs):", min_value=1, max_value=20, value=8)
sensibilidad_x = st.sidebar.slider("Sensibilidad Detección (menor = + sensible):", 0.02, 0.40, 0.15, step=0.01)

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
archivos_subidos = st.file_uploader("Arrastrá tus fotos aquí", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if archivos_subidos:
    archivos_ordenados = sorted(archivos_subidos, key=lambda x: x.name)
    num_img = len(archivos_ordenados)
    
    st.sidebar.markdown("---")
    freq_roi = st.sidebar.slider("Frec. re-cálculo ROI (cada N fotos):", 1, num_img, max(1, num_img // 2))
    w_sg_cin = st.sidebar.slider(f"Ventana Temporal (5 a {num_img}):", min_value=5, max_value=max(5, num_img), value=min(31, max(5, num_img)), step=2)
    poly_sg_cin = st.sidebar.slider("Polinomio Derivada:", 1, 5, 2)

    # =========================================================================
    # 2. CALIBRACIÓN DE PLANTILLA MAESTRA Y DETECCIÓN ESPACIAL
    # =========================================================================
    st.markdown("---")
    st.subheader("🎞️ Diagnóstico Visual (Cinta Horizontal de Referencias)")
    
    ref_indices = list(range(0, num_img, freq_roi))
    st.session_state.rois_por_ref = {}

    # --- PASE DE CALIBRACIÓN GEOMÉTRICA GLOBAL (De fotos tardías a iniciales) ---
    master_centros_x = None
    master_w_roi = 30
    master_h_roi = 20

    for i_cand in reversed(ref_indices):
        archivo_cand = archivos_ordenados[i_cand]
        archivo_cand.seek(0)
        img_bgr_cand = cv2.imdecode(np.frombuffer(archivo_cand.read(), np.uint8), cv2.IMREAD_COLOR)
        if rotacion_seleccionada is not None:
            img_bgr_cand = cv2.rotate(img_bgr_cand, rotacion_seleccionada)
        
        c_azul_c = img_bgr_cand[:, :, 0]
        p_y_c = np.mean(c_azul_c, axis=1)
        d_y_c = savgol_filter(p_y_c, window_length=w_sg_y, polyorder=poly_sg, deriv=1)
        y_min_c = int(np.argmax(d_y_c))
        y_max_c = int(np.argmin(d_y_c[y_min_c:]) + y_min_c)
        alto_banda_c = max(10, y_max_c - y_min_c)
        
        franja_c = c_azul_c[y_min_c:y_max_c, :]
        p_x_c = np.mean(franja_c, axis=0)
        d_x_c = savgol_filter(p_x_c, window_length=w_sg_x, polyorder=poly_sg, deriv=1)
        umbral_x_c = np.max(np.abs(d_x_c)) * sensibilidad_x
        
        b_izq_c, _ = find_peaks(d_x_c, height=umbral_x_c, distance=max(15, st.session_state.ancho_px // 90))
        b_der_c, _ = find_peaks(-d_x_c, height=umbral_x_c, distance=max(15, st.session_state.ancho_px // 90))
        
        cands_x = []
        for bi in b_izq_c:
            bd_cands = b_der_c[b_der_c > bi]
            if len(bd_cands) > 0:
                bd = bd_cands[0]
                if int(st.session_state.ancho_px * 0.008) < (bd - bi) < int(st.session_state.ancho_px * 0.15):
                    cands_x.append((bi, bd, (bi + bd) // 2))
        cands_x.sort(key=lambda x: x[0])
        
        if len(cands_x) == n_rois_esperado:
            master_centros_x = [c[2] for c in cands_x]
            ancho_min = min([b[1] - b[0] for b in cands_x])
            red_px = int(ancho_min * factor_reduccion * 2)
            master_w_roi = max(5, ancho_min - red_px)
            master_h_roi = max(5, alto_banda_c - red_px)
            break

    # Fallback si ninguna foto de referencia tuvo los 8 picos directos
    if master_centros_x is None:
        espaciado = st.session_state.ancho_px // (n_rois_esperado + 1)
        master_centros_x = [espaciado * (j + 1) for j in range(n_rois_esperado)]
        master_w_roi = 30
        master_h_roi = 20

    # FIJAR ÁREA DEL ROI ÁREA GLOBAL E INMUTABLE
    st.session_state.w_roi_fijo = master_w_roi
    st.session_state.h_roi_fijo = master_h_roi

    # --- PROCESAMIENTO Y ALINEACIÓN POR CONSENSO DE CADA FOTO DE REFERENCIA ---
    columnas_img = st.columns(len(ref_indices) if len(ref_indices) > 0 else 1)
    
    for idx_panel, i in enumerate(ref_indices):
        with columnas_img[idx_panel % len(columnas_img)]:
            archivo = archivos_ordenados[i]
            archivo.seek(0)
            img_bytes = archivo.read()
            img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if rotacion_seleccionada is not None: 
                img_bgr = cv2.rotate(img_bgr, rotacion_seleccionada)
            
            st.session_state.alto_px, st.session_state.ancho_px = img_bgr.shape[:2]
            canal_azul = img_bgr[:, :, 0]
            
            # Detección Y (Banda horizontal de los tubos)
            perfil_y = np.mean(canal_azul, axis=1)
            derivada_y = savgol_filter(perfil_y, window_length=w_sg_y, polyorder=poly_sg, deriv=1)
            y_min = int(np.argmax(derivada_y))
            y_max = int(np.argmin(derivada_y[y_min:]) + y_min)
            y_central = (y_min + y_max) // 2

            # Detección X local
            franja_azul = canal_azul[y_min:y_max, :]
            perfil_x = np.mean(franja_azul, axis=0)
            derivada_x = savgol_filter(perfil_x, window_length=w_sg_x, polyorder=poly_sg, deriv=1)
            umbral_x = np.max(np.abs(derivada_x)) * sensibilidad_x
            
            bordes_izq, _ = find_peaks(derivada_x, height=umbral_x, distance=max(15, st.session_state.ancho_px // 90))
            bordes_der, _ = find_peaks(-derivada_x, height=umbral_x, distance=max(15, st.session_state.ancho_px // 90))

            lista_centros_x = []
            for b_izq in bordes_izq:
                b_der_cands = bordes_der[bordes_der > b_izq]
                if len(b_der_cands) > 0:
                    b_der = b_der_cands[0]
                    if int(st.session_state.ancho_px * 0.008) < (b_der - b_izq) < int(st.session_state.ancho_px * 0.15):
                        lista_centros_x.append((b_izq, b_der, (b_izq + b_der) // 2))
            
            lista_centros_x.sort(key=lambda x: x[0])
            
            # --- ALINEACIÓN POR CONSENSO CON LA PLANTILLA MAESTRA ---
            centros_finales_x = []
            if len(lista_centros_x) == n_rois_esperado:
                centros_finales_x = [c[2] for c in lista_centros_x]
                estado_txt = "Detección Directa (8/8)"
            else:
                picos_locales = [c[2] for c in lista_centros_x]
                if len(picos_locales) > 0 and master_centros_x is not None:
                    diffs_master = np.diff(master_centros_x)
                    spacing = np.median(diffs_master) if len(diffs_master) > 0 else (st.session_state.ancho_px // (n_rois_esperado + 1))
                    max_drift = spacing / 2.0
                    
                    offsets = []
                    for p in picos_locales:
                        distancias = [abs(p - m) for m in master_centros_x]
                        idx_min = int(np.argmin(distancias))
                        if distancias[idx_min] < max_drift:
                            offsets.append(p - master_centros_x[idx_min])
                    
                    if len(offsets) > 0:
                        shift_global = int(np.median(offsets))
                        estado_txt = f"Alineación Relativa ({len(offsets)}/8 picos, Shift: {shift_global:+d}px)"
                    else:
                        shift_global = 0
                        estado_txt = "Proyección Maestra (Sin coincide.)"
                    
                    centros_finales_x = [int(cx + shift_global) for cx in master_centros_x]
                else:
                    centros_finales_x = master_centros_x
                    estado_txt = "Proyección Maestra (0 picos)"
            
            st.session_state.rois_por_ref[i] = (centros_finales_x, y_central)

            # Dibujar rectángulos en el diagnóstico visual
            img_disp = img_bgr.copy()
            cv2.line(img_disp, (0, y_min), (st.session_state.ancho_px, y_min), (255, 0, 0), 2)
            cv2.line(img_disp, (0, y_max), (st.session_state.ancho_px, y_max), (255, 0, 0), 2)
            
            w_fixed = st.session_state.w_roi_fijo
            h_fixed = st.session_state.h_roi_fijo
            for cx in centros_finales_x:
                cv2.rectangle(img_disp, (cx - w_fixed // 2, y_central - h_fixed // 2),
                              (cx + w_fixed // 2, y_central + h_fixed // 2), (0, 255, 0), 2)
            
            st.image(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB), caption=f"Ref {i} | {estado_txt}", use_container_width=True)

            del img_bgr, canal_azul, franja_azul, img_disp; gc.collect()

    # =========================================================================
    # 3. METADATOS (NOMBRES, MASAS Y COLORES)
    # =========================================================================
    st.markdown("---")
    st.subheader(f"🏷️ 2. Muestras, Masas (g) y Colores ({n_rois_esperado} Muestras)")
    
    colores_defecto = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for i in range(n_rois_esperado):
        c_id, c_name, c_mass, c_color = st.columns([1, 3, 2, 1])
        c_id.write(f"**ROI {i+1}**")
        c_name.text_input(f"Label {i+1}", value=f"Muestra_{i+1}", label_visibility="collapsed", key=f"name_{i}")
        c_mass.number_input(f"Mass {i+1}", value=1.0000, min_value=0.0001, step=0.0001, format="%.4f", label_visibility="collapsed", key=f"mass_{i}")
        c_color.color_picker(f"Color {i+1}", value=colores_defecto[i % len(colores_defecto)], label_visibility="collapsed", key=f"color_{i}")

    # =========================================================================
    # 4. EXTRACCIÓN PESADA
    # =========================================================================
    st.markdown("---")
    if st.button("▶️ Lanzar Extracción de Imágenes", use_container_width=True):
        barra = st.progress(0)
        
        tiempos_dt = [datetime.strptime(f.name.rsplit('.', 1)[0], "%Y-%m-%d_%H-%M-%S") for f in archivos_ordenados]
        t_ref = tiempos_dt[1] if len(tiempos_dt) > 1 else tiempos_dt[0]
        st.session_state.t_rel_min = np.array([(t - t_ref).total_seconds() / 60.0 for t in tiempos_dt])

        h_verde = np.zeros((num_img, n_rois_esperado))
        h_azul = np.zeros((num_img, n_rois_esperado))
        func_est = np.median if "Mediana" in metodo_estadistico else np.mean

        for idx, archivo in enumerate(archivos_ordenados):
            barra.progress(int((idx + 1) / num_img * 100))
            ref_idx = (idx // freq_roi) * freq_roi
            
            centros_x_act, y_cent_actual = st.session_state.rois_por_ref[ref_idx]
            
            archivo.seek(0)
            frame_bgr = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), cv2.IMREAD_COLOR)
            if rotacion_seleccionada is not None: 
                frame_bgr = cv2.rotate(frame_bgr, rotacion_seleccionada)
            
            c_a, c_v = frame_bgr[:, :, 0], frame_bgr[:, :, 1]
            w_r, h_r = st.session_state.w_roi_fijo, st.session_state.h_roi_fijo
            
            for r_idx, cx in enumerate(centros_x_act):
                x1 = max(0, cx - w_r // 2)
                x2 = min(st.session_state.ancho_px, cx + w_r // 2)
                y1 = max(0, y_cent_actual - h_r // 2)
                y2 = min(st.session_state.alto_px, y_cent_actual + h_r // 2)
                
                recorte_v = c_v[y1:y2, x1:x2]
                recorte_a = c_a[y1:y2, x1:x2]
                
                h_verde[idx, r_idx] = func_est(recorte_v) if recorte_v.size > 0 else 0.0
                h_azul[idx, r_idx] = func_est(recorte_a) if recorte_a.size > 0 else 0.0
                
            del frame_bgr; gc.collect()

        st.session_state.archivos_nombres = [f.name for f in archivos_ordenados]
        st.session_state.h_verde = h_verde
        st.session_state.h_azul = h_azul
        st.session_state.g0 = h_verde[0, :]
        st.session_state.a0 = h_azul[0, :]
        st.session_state.num_img = num_img
        st.session_state.procesado = True
        st.success("Extracción completada. Podés ajustar cutoff, colores y masas dinámicamente.")

    # =========================================================================
    # 5. MATEMÁTICA Y MATRIZ INTERACTIVA
    # =========================================================================
    if st.session_state.procesado:
        t = st.session_state.t_rel_min
        num_img = st.session_state.num_img
        n_r = n_rois_esperado

        st.markdown("---")
        st.subheader("⏳ 3. Control de Cutoff de Sedimentación")
        t_cutoff = st.slider("Tiempo de corte Cutoff (minutos):", 0.0, float(np.max(t)) if np.max(t) > 0 else 10.0, 2.0, step=0.5)

        g_crudo = st.session_state.h_verde
        a_crudo = st.session_state.h_azul
        r_crudo = np.where(a_crudo == 0, 1e-6, g_crudo / a_crudo)

        g_norm = g_crudo - st.session_state.g0
        a_norm = a_crudo - st.session_state.a0
        r0 = np.where(st.session_state.a0 == 0, 1e-6, st.session_state.g0 / st.session_state.a0)
        r_norm = r_crudo / r0 

        mask_cutoff = t >= t_cutoff
        t_filt = t[mask_cutoff]
        dt_prom = np.mean(np.diff(t)) if len(t) > 1 else 1.0

        w_cin = w_sg_cin if w_sg_cin <= num_img else (num_img if num_img % 2 != 0 else num_img - 1)
        poly_cin = poly_sg_cin if poly_sg_cin < w_cin else w_cin - 1

        v_gnorm, v_anorm, v_rnorm = np.full_like(g_norm, np.nan), np.full_like(a_norm, np.nan), np.full_like(r_norm, np.nan)
        v_gnorm_m, v_anorm_m, v_rnorm_m = np.full_like(g_norm, np.nan), np.full_like(a_norm, np.nan), np.full_like(r_norm, np.nan)

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
                
                masa_actual = st.session_state[f"mass_{r}"]
                v_gnorm_m[:, r] = v_gnorm[:, r] / masa_actual
                v_anorm_m[:, r] = v_anorm[:, r] / masa_actual
                v_rnorm_m[:, r] = v_rnorm[:, r] / masa_actual

        def crear_figura_plotly(x_data, y_matrix, titulo, y_label, cutoff_val=None):
            fig = go.Figure()
            for r in range(n_r):
                nombre_actual = st.session_state[f"name_{r}"]
                color_actual = st.session_state[f"color_{r}"]
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_matrix[:, r], 
                    mode='lines+markers', 
                    name=nombre_actual,
                    line=dict(width=1.5, color=color_actual),
                    marker=dict(size=4, color=color_actual)
                ))
            if cutoff_val is not None:
                fig.add_vline(x=cutoff_val, line_dash="dash", line_color="red", annotation_text="Cutoff")
            
            fig.update_layout(
                title=dict(text=titulo, font=dict(size=12)),
                xaxis_title="Tiempo (min)", yaxis_title=y_label,
                margin=dict(l=20, r=20, t=35, b=20), height=320,
                legend=dict(font=dict(size=14), orientation="h", y=-0.25),
                hovermode="x unified"
            )
            return fig

        st.markdown("---")
        st.subheader("📊 4. Panel de Gráficas")

        fig_1a = crear_figura_plotly(t, g_crudo, "1A) Verde Crudo (G)", "Intensidad", t_cutoff)
        fig_1b = crear_figura_plotly(t, a_crudo, "1B) Azul Crudo (A)", "Intensidad", t_cutoff)
        fig_1c = crear_figura_plotly(t, r_crudo, "1C) Ratio Crudo (G/A)", "Ratio (G/A)", t_cutoff)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(fig_1a, use_container_width=True)
        with c2: st.plotly_chart(fig_1b, use_container_width=True)
        with c3: st.plotly_chart(fig_1c, use_container_width=True)

        fig_2a = crear_figura_plotly(t, g_norm, "2A) Verde Neto (G - G0)", "Δ Intensidad")
        fig_2b = crear_figura_plotly(t, a_norm, "2B) Azul Neto (A - A0)", "Δ Intensidad")
        fig_2c = crear_figura_plotly(t, r_norm, "2C) Ratio Normalizado (R / R0)", "Ratio (R/R0)")

        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(fig_2a, use_container_width=True)
        with c2: st.plotly_chart(fig_2b, use_container_width=True)
        with c3: st.plotly_chart(fig_2c, use_container_width=True)

        fig_3a = crear_figura_plotly(t, v_gnorm, "3A) Vel. Verde Neto", "d(G-G0)/dt")
        fig_3b = crear_figura_plotly(t, v_anorm, "3B) Vel. Azul Neto", "d(A-A0)/dt")
        fig_3c = crear_figura_plotly(t, v_rnorm, "3C) Vel. Ratio (R/R0)", "d(R/R0)/dt")

        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(fig_3a, use_container_width=True)
        with c2: st.plotly_chart(fig_3b, use_container_width=True)
        with c3: st.plotly_chart(fig_3c, use_container_width=True)

        fig_4a = crear_figura_plotly(t, v_gnorm_m, "4A) Vel. Verde / Masa", "Unidades / (min·g)")
        fig_4b = crear_figura_plotly(t, v_anorm_m, "4B) Vel. Azul / Masa", "Unidades / (min·g)")
        fig_4c = crear_figura_plotly(t, v_rnorm_m, "4C) Vel. Ratio (R/R0) / Masa", "Unidades / (min·g)")

        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(fig_4a, use_container_width=True)
        with c2: st.plotly_chart(fig_4b, use_container_width=True)
        with c3: st.plotly_chart(fig_4c, use_container_width=True)

        # =========================================================================
        # 6. EXPORTACIONES
        # =========================================================================
        st.markdown("---")
        st.subheader("🌐 5. Exportaciones")
        
        matriz_figuras = [
            [fig_1a, fig_1b, fig_1c], 
            [fig_2a, fig_2b, fig_2c], 
            [fig_3a, fig_3b, fig_3c], 
            [fig_4a, fig_4b, fig_4c]
        ]
        
        html_cabecera = [
            "\x3c!DOCTYPE html\x3e",
            "\x3chtml lang='es'\x3e",
            "\x3chead\x3e",
            "    \x3cmeta charset='UTF-8'\x3e",
            "    \x3ctitle\x3eReporte Cinético FDA\x3c/title\x3e",
            "    \x3cstyle\x3e",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1 { text-align: center; color: #333; }",
            "        .row { display: flex; width: 100%; margin-bottom: 20px; }",
            "        .col { flex: 33.33%; padding: 5px; box-sizing: border-box; }",
            "    \x3c/style\x3e",
            "\x3c/head\x3e",
            "\x3cbody\x3e",
            "    \x3ch1\x3e📊 Reporte Cinético FDA - Matriz 3x4\x3c/h1\x3e"
        ]
        
        html_content = "\n".join(html_cabecera) + "\n"
        for i, fila in enumerate(matriz_figuras):
            html_content += "\x3cdiv class='row'\x3e\n"
            for fig in fila:
                use_cdn = "cdn" if i == 0 else False
                fig_html = fig.to_html(full_html=False, include_plotlyjs=use_cdn)
                html_content += f"\x3cdiv class='col'\x3e{fig_html}\x3c/div\x3e\n"
            html_content += "\x3c/div\x3e\n"
            
        html_content += "\x3c/body\x3e\n\x3c/html\x3e\n"
        
        st.download_button(
            "📥 Descargar Reporte Interactivo (HTML)", 
            html_content.encode('utf-8'), 
            f"Reporte_FDA_{int(time.time())}.html", 
            "text/html"
        )

        cols = ["Archivo", "Tiempo_Min"]
        for r in range(n_r):
            nm = st.session_state[f"name_{r}"]
            cols.extend([f"{nm}_G", f"{nm}_A", f"{nm}_R", f"{nm}_G-G0", f"{nm}_A-A0", f"{nm}_R/R0", f"{nm}_vG", f"{nm}_vA", f"{nm}_v(R/R0)", f"{nm}_vG_m", f"{nm}_vA_m", f"{nm}_v(R/R0)_m", f"{nm}_Masa"])
            
        datos = []
        for i_img in range(num_img):
            f = [st.session_state.archivos_nombres[i_img], t[i_img]]
            for r in range(n_r):
                f.extend([
                    g_crudo[i_img, r], a_crudo[i_img, r], r_crudo[i_img, r], 
                    g_norm[i_img, r], a_norm[i_img, r], r_norm[i_img, r], 
                    v_gnorm[i_img, r], v_anorm[i_img, r], v_rnorm[i_img, r], 
                    v_gnorm_m[i_img, r], v_anorm_m[i_img, r], v_rnorm_m[i_img, r], 
                    st.session_state[f"mass_{r}"]
                ])
            datos.append(f)

        df_exp = pd.DataFrame(datos, columns=cols)
        st.download_button("📥 Descargar Tabla (CSV)", df_exp.to_csv(index=False).encode('utf-8'), f"fda_data_{int(time.time())}.csv", "text/csv")
