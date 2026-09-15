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
st.set_page_config(page_title="Análisis FDA - Alta Eficiencia", layout="wide")
st.title("🔬 Análisis FDA - Matriz 3x4 Interactiva")

if "procesado" not in st.session_state:
    st.session_state.procesado = False

# =========================================================================
# FUNCIONES CACHEADAS (EVITAN RE-PROCESAR IMÁGENES)
# =========================================================================
@st.cache_resource(show_spinner="Procesando detección espacial de ROIs...")
def procesar_diagnostico_rois(archivos_subidos, opcion_rotar, prop_sg_y, prop_sg_x, poly_sg, factor_reduccion, freq_roi):
    """
    Decodifica imágenes y calcula las coordenadas de ROIs solo cuando
    cambian los parámetros espaciales o las imágenes cargadas.
    """
    dict_rotacion = {
        "Sin Rotación": None, 
        "180 Grados": cv2.ROTATE_180, 
        "90 Grados Horario": cv2.ROTATE_90_CLOCKWISE, 
        "90 Grados Antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    rot_sel = dict_rotacion[opcion_rotar]
    
    archivos_ordenados = sorted(archivos_subidos, key=lambda x: x.name)
    num_img = len(archivos_ordenados)
    ref_indices = list(range(0, num_img, freq_roi))
    
    rois_por_ref = {}
    imagenes_preview = []
    n_rois_base = 0
    w_roi_fijo, h_roi_fijo = 30, 20

    # Obtener dimensiones base con la primera imagen
    bytes_0 = archivos_ordenados[0].getvalue()
    img_0 = cv2.imdecode(np.frombuffer(bytes_0, np.uint8), cv2.IMREAD_COLOR)
    if rot_sel is not None: img_0 = cv2.rotate(img_0, rot_sel)
    alto_px, ancho_px = img_0.shape[:2]

    w_sg_y = max(5, int((prop_sg_y / 100.0) * alto_px))
    if w_sg_y % 2 == 0: w_sg_y += 1
    w_sg_x = max(5, int((prop_sg_x / 100.0) * ancho_px))
    if w_sg_x % 2 == 0: w_sg_x += 1

    for i in ref_indices:
        archivo = archivos_ordenados[i]
        img_bytes = archivo.getvalue()
        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if rot_sel is not None: img_bgr = cv2.rotate(img_bgr, rot_sel)
        
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
        bordes_izq, _ = find_peaks(derivada_x, height=umbral_x, distance=max(15, ancho_px // 90))
        bordes_der, _ = find_peaks(-derivada_x, height=umbral_x, distance=max(15, ancho_px // 90))

        lista_centros_x = []
        for b_izq in bordes_izq:
            b_der_cands = bordes_der[bordes_der > b_izq]
            if len(b_der_cands) > 0:
                b_der = b_der_cands[0]
                if int(ancho_px * 0.008) < (b_der - b_izq) < int(ancho_px * 0.08):
                    lista_centros_x.append((b_izq, b_der, (b_izq + b_der) // 2))

        lista_centros_x.sort(key=lambda x: x[0])
        
        if i == 0:
            n_rois_base = len(lista_centros_x)
            if n_rois_base > 0:
                ancho_min = min([b[1] - b[0] for b in lista_centros_x])
                red_px = int(ancho_min * factor_reduccion * 2)
                w_roi_fijo, h_roi_fijo = ancho_min - red_px, alto_banda - red_px
            rois_por_ref[i] = (lista_centros_x, y_central)
        else:
            if len(lista_centros_x) != n_rois_base:
                lista_centros_x, y_central = rois_por_ref[0]
            rois_por_ref[i] = (lista_centros_x, y_central)

        img_disp = img_bgr.copy()
        cv2.line(img_disp, (0, y_min), (ancho_px, y_min), (255, 0, 0), 2)
        cv2.line(img_disp, (0, y_max), (ancho_px, y_max), (255, 0, 0), 2)
        for _, _, cx in lista_centros_x:
            cv2.rectangle(img_disp, (cx - w_roi_fijo // 2, y_central - h_roi_fijo // 2),
                          (cx + w_roi_fijo // 2, y_central + h_roi_fijo // 2), (0, 255, 0), 2)
        
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        imagenes_preview.append((f"Ref {i} ({len(lista_centros_x)} ROIs)", img_rgb))

    return rois_por_ref, imagenes_preview, n_rois_base, w_roi_fijo, h_roi_fijo, ancho_px, alto_px

# =========================================================================
# CONTROLES DE PARÁMETROS (BARRA LATERAL)
# =========================================================================
st.sidebar.header("⚙️ Configuración Espacial")
opcion_rotar = st.sidebar.selectbox("Rotación de Cámara:", ["Sin Rotación", "180 Grados", "90 Grados Horario", "90 Grados Antihorario"])
metodo_estadistico = st.sidebar.radio("Cálculo de Intensidad:", ["Mediana", "Promedio"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Filtros S-G: Detección Espacial")
prop_sg_y = st.sidebar.slider("Ventana S-G Y (%):", 0.5, 15.0, 1.5, step=0.1)
prop_sg_x = st.sidebar.slider("Ventana S-G X (%):", 0.1, 10.0, 0.5, step=0.1)
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
    st.sidebar.subheader("📈 Filtros Cinemáticos Temporales")
    freq_roi = st.sidebar.slider("Frec. re-cálculo ROI (cada N fotos):", 1, num_img, 1)
    w_sg_cin = st.sidebar.slider(f"Ventana Temporal (5 a {num_img}):", min_value=5, max_value=max(5, num_img), value=min(31, max(5, num_img)), step=2)
    poly_sg_cin = st.sidebar.slider("Polinomio Derivada:", 1, 5, 2)

    # =========================================================================
    # 2. DETECCIÓN Y DIAGNÓSTICO VISUAL (EJECUCIÓN CACHEADA)
    # =========================================================================
    rois_por_ref, imagenes_preview, n_rois_base, w_roi_fijo, h_roi_fijo, ancho_px, alto_px = procesar_diagnostico_rois(
        archivos_subidos, opcion_rotar, prop_sg_y, prop_sg_x, poly_sg, factor_reduccion, freq_roi
    )

    st.markdown("---")
    st.subheader("🎞️ Diagnóstico Visual (Cinta Horizontal de Referencias)")
    
    cols_preview = st.columns(len(imagenes_preview) if len(imagenes_preview) > 0 else 1)
    for idx, (cap, img_rgb) in enumerate(imagenes_preview):
        with cols_preview[idx % len(cols_preview)]:
            st.image(img_rgb, caption=cap, use_container_width=True)

    # =========================================================================
    # 3. METADATOS (NOMBRES, MASAS Y COLORES)
    # =========================================================================
    st.markdown("---")
    st.subheader("🏷️ 2. Muestras, Masas (g) y Colores")
    
    colores_defecto = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for i in range(n_rois_base):
        c_id, c_name, c_mass, c_color = st.columns([1, 3, 2, 1])
        c_id.write(f"**ROI {i+1}**")
        c_name.text_input(f"Label {i+1}", value=f"Muestra_{i+1}", label_visibility="collapsed", key=f"name_{i}")
        c_mass.number_input(f"Mass {i+1}", value=1.0000, min_value=0.0001, step=0.0001, format="%.4f", label_visibility="collapsed", key=f"mass_{i}")
        c_color.color_picker(f"Color {i+1}", value=colores_defecto[i % len(colores_defecto)], label_visibility="collapsed", key=f"color_{i}")

    # =========================================================================
    # 4. EXTRACCIÓN HEAVY (SOLO SE EJECUTA UNA VEZ AL APRETAR EL BOTÓN)
    # =========================================================================
    st.markdown("---")
    if st.button("▶️ Lanzar Extracción de Intensidades", use_container_width=True):
        if n_rois_base > 0:
            barra = st.progress(0)
            dict_rot = {"Sin Rotación": None, "180 Grados": cv2.ROTATE_180, "90 Grados Horario": cv2.ROTATE_90_CLOCKWISE, "90 Grados Antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE}
            rot_sel = dict_rot[opcion_rotar]

            tiempos_dt = [datetime.strptime(f.name.rsplit('.', 1)[0], "%Y-%m-%d_%H-%M-%S") for f in archivos_ordenados]
            t_ref = tiempos_dt[1] if len(tiempos_dt) > 1 else tiempos_dt[0]
            st.session_state.t_rel_min = np.array([(t - t_ref).total_seconds() / 60.0 for t in tiempos_dt])

            h_verde, h_azul = np.zeros((num_img, n_rois_base)), np.zeros((num_img, n_rois_base))
            func_est = np.median if "Mediana" in metodo_estadistico else np.mean

            for idx, archivo in enumerate(archivos_ordenados):
                barra.progress(int((idx + 1) / num_img * 100))
                ref_idx = (idx // freq_roi) * freq_roi
                centros_tuplas, y_cent_actual = rois_por_ref[ref_idx]
                centros_x_act = [b[2] for b in centros_tuplas]
                
                frame_bgr = cv2.imdecode(np.frombuffer(archivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if rot_sel is not None: frame_bgr = cv2.rotate(frame_bgr, rot_sel)
                
                c_a, c_v = frame_bgr[:, :, 0], frame_bgr[:, :, 1]
                
                for r_idx, cx in enumerate(centros_x_act):
                    x1, x2 = cx - w_roi_fijo // 2, cx + w_roi_fijo // 2
                    y1, y2 = y_cent_actual - h_roi_fijo // 2, y_cent_actual + h_roi_fijo // 2
                    h_verde[idx, r_idx] = func_est(c_v[y1:y2, x1:x2])
                    h_azul[idx, r_idx] = func_est(c_a[y1:y2, x1:x2])
                del frame_bgr; gc.collect()

            st.session_state.archivos_nombres = [f.name for f in archivos_ordenados]
            st.session_state.h_verde, st.session_state.h_azul = h_verde, h_azul
            st.session_state.g0, st.session_state.a0 = h_verde[0, :], h_azul[0, :]
            st.session_state.num_img = num_img
            st.session_state.n_rois_base = n_rois_base
            st.session_state.procesado = True
            st.success("¡Extracción de datos finalizada! A partir de ahora podés cambiar Cutoff, Filtros temporales, Nombres, Masas y Colores instantáneamente.")

    # =========================================================================
    # 5. MATEMÁTICA Y MATRIZ INTERACTIVA (EN MEMORIA - MILISEGUNDOS)
    # =========================================================================
    if st.session_state.procesado:
        t = st.session_state.t_rel_min
        num_img = st.session_state.num_img
        n_r = st.session_state.n_rois_base

        st.markdown("---")
        st.subheader("⏳ 3. Control de Cutoff de Sedimentación")
        t_cutoff = st.slider("Tiempo de corte Cutoff (minutos):", 0.0, float(np.max(t)) if np.max(t) > 0 else 10.0, 2.0, step=0.5)

        # 1. Recuperar datos crudos extraídos
        g_crudo = st.session_state.h_verde
        a_crudo = st.session_state.h_azul
        r_crudo = np.where(a_crudo == 0, 1e-6, g_crudo / a_crudo)

        # 2. Normalización básica
        g_norm = g_crudo - st.session_state.g0
        a_norm = a_crudo - st.session_state.a0
        r0 = np.where(st.session_state.a0 == 0, 1e-6, st.session_state.g0 / st.session_state.a0)
        r_norm = r_crudo / r0 

        # 3. Aplicar Filtro Temporal S-G y Cutoff
        mask_cutoff = t >= t_cutoff
        dt_prom = np.mean(np.diff(t)) if len(t) > 1 else 1.0

        w_cin = w_sg_cin if w_sg_cin <= num_img else (num_img if num_img % 2 != 0 else num_img - 1)
        poly_cin = poly_sg_cin if poly_sg_cin < w_cin else w_cin - 1

        v_gnorm, v_anorm, v_rnorm = np.full_like(g_norm, np.nan), np.full_like(a_norm, np.nan), np.full_like(r_norm, np.nan)
        v_gnorm_m, v_anorm_m, v_rnorm_m = np.full_like(g_norm, np.nan), np.full_like(a_norm, np.nan), np.full_like(r_norm, np.nan)

        if np.sum(mask_cutoff) >= w_cin:
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
                
                # Obtener masa actual dinámica del formulario
                masa_actual = st.session_state[f"mass_{r}"]
                v_gnorm_m[:, r] = v_gnorm[:, r] / masa_actual
                v_anorm_m[:, r] = v_anorm[:, r] / masa_actual
                v_rnorm_m[:, r] = v_rnorm[:, r] / masa_actual

        def crear_figura_plotly(x_data, y_matrix, titulo, y_label, cutoff_val=None):
            fig = go.Figure()
            for r in range(n_r):
                nombre_act = st.session_state[f"name_{r}"]
                color_act = st.session_state[f"color_{r}"]
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_matrix[:, r], 
                    mode='lines+markers', 
                    name=nombre_act,
                    line=dict(width=1.5, color=color_act),
                    marker=dict(size=4, color=color_act)
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
        
        # Usamos \x3c y \x3e (que en Python son < y >) para evitar bugs visuales.
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
