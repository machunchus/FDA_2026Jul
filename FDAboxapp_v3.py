import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import gc
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

# =========================================================================
# CONFIGURACIÓN E INICIALIZACIÓN DEL ESTADO
# =========================================================================
st.set_page_config(page_title="Análisis FDA - Sensor de Suelo v4", layout="wide")

st.title("🔬 Herramienta de Análisis FDA v4 (Sensor de Suelo)")
st.write("Motor de procesamiento cinético con detección adaptativa y tracking por equidistancia.")

if "procesado" not in st.session_state:
    st.session_state.procesado = False
    st.session_state.df_resultados = None
    st.session_state.num_rois = 0
    st.session_state.num_imagenes = 0
    st.session_state.tiempos_min = None
    st.session_state.historico_verde = None
    st.session_state.historico_azul = None
    st.session_state.historico_ratios = None
    st.session_state.historico_velocidades = None
    st.session_state.vel_verde = None
    st.session_state.vel_azul = None
    st.session_state.lista_centros_x = []
    st.session_state.y_min = 0
    st.session_state.y_max = 0
    st.session_state.y_central = 0
    st.session_state.ancho_roi_final = 0
    st.session_state.alto_roi_final = 0
    st.session_state.estrategia_actual = "picos"

# =========================================================================
# FUNCIONES AUXILIARES
# =========================================================================
def extraer_tiempos_de_nombres(nombres_archivos):
    tiempos_seg = []
    for nombre in nombres_archivos:
        match = re.search(r'(\d{2})-(\d{2})-(\d{2})', nombre)
        if match:
            h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
            tiempos_seg.append(h * 3600 + m * 60 + s)
        else:
            tiempos_seg.append(0)
    
    if len(tiempos_seg) > 0:
        t0 = tiempos_seg[0]
        tiempos_seg_norm = [t - t0 for t in tiempos_seg]
        tiempos_min = [t / 60.0 for t in tiempos_seg_norm]
    else:
        tiempos_min = []
    return tiempos_min

def detectar_por_equidistancia(canal_azul, y_min, y_max, n_rois_esperado, ancho_px):
    franja = canal_azul[y_min:y_max, :]
    perfil_x = np.mean(franja, axis=0)
    espaciado = ancho_px // (n_rois_esperado + 1)
    centros = []
    for i in range(n_rois_esperado):
        cx_esperado = espaciado * (i + 1)
        margen = int(espaciado * 0.2)
        x_ini_busq = max(0, cx_esperado - margen)
        x_fin_busq = min(ancho_px, cx_esperado + margen)
        region = perfil_x[x_ini_busq:x_fin_busq]
        if len(region) > 0:
            cx_real = x_ini_busq + np.argmax(region)
            ancho_estimado = int(espaciado * 0.6)
            x_ini_roi = max(0, cx_real - ancho_estimado // 2)
            x_fin_roi = min(ancho_px, cx_real + ancho_estimado // 2)
            centros.append((x_ini_roi, x_fin_roi, cx_real))
    return centros

# =========================================================================
# BARRA LATERAL (CONTROLES)
# =========================================================================
st.sidebar.header("⚙️ Configuración del Análisis")

opcion_rotar = st.sidebar.selectbox("Rotación de Cámara:", ["Sin Rotación", "180 Grados", "90 Grados Horario", "90 Grados Antihorario"])
dict_rotacion = {"Sin Rotación": None, "180 Grados": cv2.ROTATE_180, "90 Grados Horario": cv2.ROTATE_90_CLOCKWISE, "90 Grados Antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE}
rotacion_seleccionada = dict_rotacion[opcion_rotar]

st.sidebar.markdown("---")
frecuencia_roi = st.sidebar.slider("Frecuencia de re-tracking (cada x fotogramas):", 1, 50, 10)

if "ancho_px" not in st.session_state: st.session_state.ancho_px = 1280
if "alto_px" not in st.session_state: st.session_state.alto_px = 960

st.sidebar.markdown("---")
prop_sg_y = st.sidebar.slider("Ventana S-G en Y (%):", 0.5, 15.0, 5.0, step=0.1)
w_sg_y = int((prop_sg_y / 100.0) * st.session_state.alto_px)
w_sg_y = w_sg_y if w_sg_y % 2 != 0 else w_sg_y + 1
w_sg_y = max(5, w_sg_y)

prop_sg_x = st.sidebar.slider("Ventana S-G en X (%):", 0.5, 10.0, 1.5, step=0.1)
w_sg_x = int((prop_sg_x / 100.0) * st.session_state.ancho_px)
w_sg_x = w_sg_x if w_sg_x % 2 != 0 else w_sg_x + 1
w_sg_x = max(5, w_sg_x)

poly_sg = st.sidebar.slider("Orden Polinomio Detección:", 2, 5, 4)
factor_reduccion = st.sidebar.slider("Factor reducción ROI:", 0.0, 0.45, 0.20, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Detección de ROIs")
sensibilidad_x = st.sidebar.slider("Sensibilidad detección X (menor = más sensible):", 0.02, 0.30, 0.15, step=0.01)
n_rois_esperado = st.sidebar.number_input("N° esperado de ROIs:", 1, 20, 8)

estrategia = st.sidebar.radio("Estrategia de detección:", ["Detección por picos (S-G)", "Detección por equidistancia"], index=0)
st.session_state.estrategia_actual = "picos" if estrategia == "Detección por picos (S-G)" else "equidistancia"

st.sidebar.markdown("---")
w_sg_cin = st.sidebar.slider("Ventana S-G Temporal (Velocidad):", 5, 41, 11, step=2)
poly_sg_cin = st.sidebar.slider("Orden Polinomio Velocidad:", 2, 5, 2)

# =========================================================================
# SECCIÓN 1 Y 2: CARGA Y CALIBRACIÓN
# =========================================================================
st.subheader("🗂️ 1. Carga de Imágenes Secuenciales")
archivos_subidos = st.file_uploader("Arrastrá tus fotos aquí", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if archivos_subidos:
    archivos_ordenados = sorted(archivos_subidos, key=lambda x: x.name)
    
    st.markdown("---")
    st.subheader(" 2. Diagnóstico Visual y Calibración")
    
    try:
        mejor_score = 0
        mejor_deteccion = None
        mejor_idx = -1
        
        indices_evaluacion = [0]
        if len(archivos_ordenados) > 1:
            indices_evaluacion.append(len(archivos_ordenados) - 1)
        if len(archivos_ordenados) > 10:
            indices_evaluacion.append(len(archivos_ordenados) // 2)
        
        for idx_eval in indices_evaluacion:
            archivo_eval = archivos_ordenados[idx_eval]
            img_bytes = archivo_eval.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            archivo_eval.seek(0)
            
            if rotacion_seleccionada is not None:
                img_bgr = cv2.rotate(img_bgr, rotacion_seleccionada)
            
            alto_px, ancho_px = img_bgr.shape[:2]
            canal_azul = img_bgr[:, :, 0]
            
            perfil_y = np.mean(canal_azul, axis=1)
            derivada_y = savgol_filter(perfil_y, window_length=w_sg_y, polyorder=poly_sg, deriv=1)
            y_min = int(np.argmax(derivada_y))
            y_max = int(np.argmin(derivada_y[y_min:]) + y_min)
            y_central = (y_min + y_max) // 2
            alto_banda = y_max - y_min
            
            if st.session_state.estrategia_actual == "equidistancia":
                lista_centros_x = detectar_por_equidistancia(canal_azul, y_min, y_max, n_rois_esperado, ancho_px)
            else:
                franja_azul = canal_azul[y_min:y_max, :]
                perfil_x = np.mean(franja_azul, axis=0)
                derivada_x = savgol_filter(perfil_x, window_length=w_sg_x, polyorder=poly_sg, deriv=1)
                
                umbral_x = np.max(np.abs(derivada_x)) * sensibilidad_x
                min_dist_x = max(15, ancho_px // 90)
                bordes_izq, _ = find_peaks(derivada_x, height=umbral_x, distance=min_dist_x)
                bordes_der, _ = find_peaks(-derivada_x, height=umbral_x, distance=min_dist_x)
                
                lista_centros_x = []
                for b_izq in bordes_izq:
                    posibles_derechos = bordes_der[bordes_der > b_izq]
                    if len(posibles_derechos) > 0:
                        b_der = posibles_derechos[0]
                        if int(ancho_px * 0.008) < (b_der - b_izq) < int(ancho_px * 0.15):
                            lista_centros_x.append((b_izq, b_der, (b_izq + b_der) // 2))
                lista_centros_x.sort(key=lambda x: x[0])
            
            score = len(lista_centros_x)
            if len(lista_centros_x) == n_rois_esperado:
                score += 100
            
            if score > mejor_score:
                mejor_score = score
                mejor_idx = idx_eval
                
                if len(lista_centros_x) > 0:
                    ancho_minimo = min([b[1] - b[0] for b in lista_centros_x])
                    reduccion_px_total = int(ancho_minimo * factor_reduccion * 2)
                    ancho_roi_final = ancho_minimo - reduccion_px_total
                    alto_roi_final = alto_banda - reduccion_px_total
                else:
                    ancho_roi_final, alto_roi_final = 30, 20
                
                mejor_deteccion = (lista_centros_x, y_min, y_max, y_central, ancho_roi_final, alto_roi_final)
        
        if mejor_deteccion is not None:
            lista_centros_x, y_min, y_max, y_central, ancho_roi_final, alto_roi_final = mejor_deteccion
            st.info(f"✅ Mejor detección encontrada en imagen {mejor_idx + 1}: {len(lista_centros_x)} ROIs")
        else:
            st.error("❌ No se pudo detectar ningún ROI en ninguna imagen de referencia")
            st.stop()
        
        st.session_state.lista_centros_x = lista_centros_x
        st.session_state.y_min = y_min
        st.session_state.y_max = y_max
        st.session_state.y_central = y_central
        st.session_state.ancho_roi_final = ancho_roi_final
        st.session_state.alto_roi_final = alto_roi_final
        
        archivo_eval = archivos_ordenados[mejor_idx]
        img_bytes = archivo_eval.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        archivo_eval.seek(0)
        if rotacion_seleccionada is not None:
            img_bgr = cv2.rotate(img_bgr, rotacion_seleccionada)
        
        img_mascara = img_bgr.copy()
        for i, (b_izq, b_der, cx) in enumerate(lista_centros_x):
            cv2.rectangle(img_mascara, (b_izq, y_min), (b_der, y_max), (0, 255, 0), 2)
            x1_roi, x2_roi = cx - ancho_roi_final // 2, cx + ancho_roi_final // 2
            y1_roi, y2_roi = y_central - alto_roi_final // 2, y_central + alto_roi_final // 2
            cv2.rectangle(img_mascara, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 215, 255), 1)
            cv2.circle(img_mascara, (cx, y_central), 3, (0, 0, 255), -1)
            cv2.putText(img_mascara, f"R_{i+1}", (b_izq, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        st.image(img_mascara, channels="BGR", caption=f"Calibración Activa: {len(lista_centros_x)} capilares detectados (estrategia: {st.session_state.estrategia_actual}).", use_container_width=True)
        
        del img_bgr, img_mascara, canal_azul
        gc.collect()
        
    except Exception as e:
        st.error(f"Error crítico en calibración: {e}")
        st.stop()

    # =========================================================================
    # SECCIÓN 3: PROCESAMIENTO
    # =========================================================================
    st.markdown("---")
    st.subheader("🚀 3. Procesamiento General de Lote")
    
    if st.button("▶️ Ejecutar Análisis de Lote Semiatomático"):
        if len(st.session_state.lista_centros_x) > 0:
            barra_progreso = st.progress(0)
            texto_estado = st.empty()
            
            num_img = len(archivos_ordenados)
            n_rois = len(st.session_state.lista_centros_x)
            
            h_verde = np.zeros((num_img, n_rois))
            h_azul = np.zeros((num_img, n_rois))
            h_ratios = np.zeros((num_img, n_rois))
            
            centros_din_x = [b[2] for b in st.session_state.lista_centros_x]
            y_min_din = st.session_state.y_min
            y_max_din = st.session_state.y_max
            y_cen_din = st.session_state.y_central
            w_roi = st.session_state.ancho_roi_final
            h_roi = st.session_state.alto_roi_final

            nombres_archivos = [f.name for f in archivos_ordenados]
            tiempos_min = extraer_tiempos_de_nombres(nombres_archivos)
            
            st.info(f"⏱️ Tiempos extraídos: {tiempos_min[-1]:.1f} minutos totales")

            for idx, archivo in enumerate(archivos_ordenados):
                barra_progreso.progress(int((idx + 1) / num_img * 100))
                texto_estado.text(f"Analizando {idx + 1}/{num_img}: {archivo.name}")
                
                img_bytes = archivo.read()
                frame_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                archivo.seek(0)
                if rotacion_seleccionada is not None: 
                    frame_bgr = cv2.rotate(frame_bgr, rotacion_seleccionada)
                    
                c_azul_f = frame_bgr[:, :, 0]
                c_verde_f = frame_bgr[:, :, 1]
                
                if idx > 0 and idx % frecuencia_roi == 0:
                    p_y_f = np.mean(c_azul_f, axis=1)
                    d_y_f = savgol_filter(p_y_f, window_length=w_sg_y, polyorder=poly_sg, deriv=1)
                    y_min_din = int(np.argmax(d_y_f))
                    y_max_din = int(np.argmin(d_y_f[y_min_din:]) + y_min_din)
                    y_cen_din = (y_min_din + y_max_din) // 2
                
                for r_idx, cx_din in enumerate(centros_din_x):
                    x1, x2 = cx_din - w_roi // 2, cx_din + w_roi // 2
                    y1, y2 = y_cen_din - h_roi // 2, y_cen_din + h_roi // 2
                    
                    mean_v = np.mean(c_verde_f[y1:y2, x1:x2])
                    mean_a = np.mean(c_azul_f[y1:y2, x1:x2])
                    h_verde[idx, r_idx] = mean_v
                    h_azul[idx, r_idx] = mean_a
                    h_ratios[idx, r_idx] = mean_v / (mean_a if mean_a > 0 else 1.0)
                
                del frame_bgr, c_azul_f, c_verde_f, img_bytes
                if idx % 10 == 0:
                    gc.collect()

            w_cin_valida = w_sg_cin if w_sg_cin <= num_img else num_img
            if w_cin_valida % 2 == 0: w_cin_valida -= 1
            poly_cin_valido = poly_sg_cin if poly_sg_cin < w_cin_valida else w_cin_valida - 1
            if poly_cin_valido < 1: poly_cin_valido = 1
            
            h_velocidades = np.zeros_like(h_ratios)
            vel_verde = np.zeros_like(h_verde)
            vel_azul = np.zeros_like(h_azul)
            
            for r in range(n_rois):
                h_velocidades[:, r] = savgol_filter(h_ratios[:, r], window_length=w_cin_valida, polyorder=poly_cin_valido, deriv=1)
                vel_verde[:, r] = savgol_filter(h_verde[:, r], window_length=w_cin_valida, polyorder=poly_cin_valido, deriv=1)
                vel_azul[:, r] = savgol_filter(h_azul[:, r], window_length=w_cin_valida, polyorder=poly_cin_valido, deriv=1)

            columnas_df = ["Fotograma", "Nombre_Archivo", "Tiempo_min"]
            for r in range(n_rois):
                columnas_df.extend([
                    f"R{r+1}_Verde", f"R{r+1}_Azul", f"R{r+1}_Ratio",
                    f"R{r+1}_Vel_Verde", f"R{r+1}_Vel_Azul", f"R{r+1}_Vel_Ratio"
                ])
                
            datos_totales = []
            for i in range(num_img):
                fila = [i, nombres_fotos[i], tiempos_min[i]]
                for r in range(n_rois):
                    fila.extend([
                        h_verde[i, r], h_azul[i, r], h_ratios[i, r],
                        vel_verde[i, r], vel_azul[i, r], h_velocidades[i, r]
                    ])
                datos_totales.append(fila)

            st.session_state.df_resultados = pd.DataFrame(datos_totales, columns=columnas_df)
            st.session_state.num_rois = n_rois
            st.session_state.num_imagenes = num_img
            st.session_state.tiempos_min = tiempos_min
            st.session_state.historico_verde = h_verde
            st.session_state.historico_azul = h_azul
            st.session_state.historico_ratios = h_ratios
            st.session_state.historico_velocidades = h_velocidades
            st.session_state.vel_verde = vel_verde
            st.session_state.vel_azul = vel_azul
            st.session_state.procesado = True
        else:
            st.error("No hay capilares calibrados.")

    # =========================================================================
    # SECCIÓN 4: RENDERIZADO RESULTADOS
    # =========================================================================
    if st.session_state.procesado:
        st.markdown("---")
        st.subheader("📊 Reporte de Dinámicas Espectrales")
        
        col_g1, col_g2, col_g3 = st.columns(3)
        t = st.session_state.tiempos_min
        n_rois = st.session_state.num_rois

        plt.close('all')

        with col_g1:
            st.write("**Intensidad Verde vs Tiempo**")
            fig_v, ax_v = plt.subplots(figsize=(5, 4))
            for r in range(n_rois): ax_v.plot(t, st.session_state.historico_verde[:, r])
            ax_v.set_xlabel("Tiempo (min)"); ax_v.set_ylabel("Intensidad Verde")
            st.pyplot(fig_v)

        with col_g2:
            st.write("**Intensidad Azul vs Tiempo**")
            fig_a, ax_a = plt.subplots(figsize=(5, 4))
            for r in range(n_rois): ax_a.plot(t, st.session_state.historico_azul[:, r])
            ax_a.set_xlabel("Tiempo (min)"); ax_a.set_ylabel("Intensidad Azul")
            st.pyplot(fig_a)

        with col_g3:
            st.write("**Ratio (G/B) vs Tiempo**")
            fig_r, ax_r = plt.subplots(figsize=(5, 4))
            for r in range(n_rois): ax_r.plot(t, st.session_state.historico_ratios[:, r])
            ax_r.set_xlabel("Tiempo (min)"); ax_r.set_ylabel("Ratio Verde/Azul")
            st.pyplot(fig_r)

        st.markdown("---")
        st.subheader("📈 Perfiles de Velocidad (1ª Derivada S-G)")
        
        col_v1, col_v2, col_v3 = st.columns(3)
        
        with col_v1:
            st.write("**Velocidad Verde (dV/dt)**")
            fig_vv, ax_vv = plt.subplots(figsize=(5, 4))
            for r in range(n_rois): ax_vv.plot(t, st.session_state.vel_verde[:, r], label=f"R{r+1}", linewidth=1.2)
            ax_vv.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_vv.set_xlabel("Tiempo (min)"); ax_vv.set_ylabel("Velocidad Verde")
            ax_vv.legend(fontsize=7, ncol=2)
            st.pyplot(fig_vv)

        with col_v2:
            st.write("**Velocidad Azul (dB/dt)**")
            fig_va, ax_va = plt.subplots(figsize=(5, 4))
            for r in range(n_rois): ax_va.plot(t, st.session_state.vel_azul[:, r], label=f"R{r+1}", linewidth=1.2)
            ax_va.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_va.set_xlabel("Tiempo (min)"); ax_va.set_ylabel("Velocidad Azul")
            ax_va.legend(fontsize=7, ncol=2)
            st.pyplot(fig_va)

        with col_v3:
            st.write("**Velocidad Ratio (d(G/B)/dt)**")
            fig_vr, ax_vr = plt.subplots(figsize=(5, 4))
            for r in range(n_rois): ax_vr.plot(t, st.session_state.historico_velocidades[:, r], label=f"R{r+1}", linestyle="--", linewidth=1.5)
            ax_vr.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_vr.set_xlabel("Tiempo (min)"); ax_vr.set_ylabel("Velocidad Ratio")
            ax_vr.legend(fontsize=7, ncol=2)
            st.pyplot(fig_vr)

        plt.close('all')

        st.markdown("---")
        st.subheader("💾 Exportación de Datos")
        st.dataframe(st.session_state.df_resultados.head(5).style.format(precision=4), use_container_width=True)
        
        csv_utf8_bytes = st.session_state.df_resultados.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=" Descargar CSV Completo",
            data=csv_utf8_bytes,
            file_name=f"reporte_FDA_{int(time.time())}.csv",
            mime="text/csv",
            key=f"dl_btn_{int(time.time())}"
        )
