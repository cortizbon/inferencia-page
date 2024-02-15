import streamlit as st
import numpy as np
from scipy.stats import t, norm, f
import matplotlib.pyplot as plt

st.title("Estadística inferencial")

tab1, tab2 = st.tabs(["Intervalos de confianza",
                      "Pruebas de hipótesis"])

with tab1:
    # generar una distribución t
    x = np.linspace(-3, 3, 10000)

    st.subheader("Media")

    st.write("Intervalo para las medias: ")

    st.latex(r"\bar{x}\pm t_{\alpha/2}\frac{s}{\sqrt{n}}")

    st.write("Intervalo para la diferencia de medias con varianzas iguales: ")

    st.latex(r"\bar{x}_1-\bar{x}_2 \pm t_{\alpha/2}S_p \sqrt{\frac{1}{n_1}+\frac{1}{n_2}}")
    st.latex(r"S^2_P=\frac{(n_1-1)S_1^2+(n_2-1)S^2_2}{n_1+n_2-2}")

    st.write("Intervalo para la diferencia de medias con varianzas diferentes: ")

    st.latex(r"\bar{x}_1-\bar{x}_2 \pm t_{\alpha/2} \sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}")

    st.code("=DISTR.T.INV(alpha/2; df)", language='excelFormula')

    T = t(30)
    y = T.pdf(x)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.plot(x, y)
    ax.set_title('t')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)

    st.divider()

    st.subheader("Varianza")

    st.write("Intervalo para cociente de varianzas: ")
    

    st.latex(r"\frac{\sigma^2_1}{\sigma^2_2} \pm\left(\frac{s_1^2}{s_2^2}\right)F_{\alpha/2}(n_1-1, n_2-1) ")

    st.code("=DISTR.F.INV(1-alpha;df_num;df_den)", language='excelFormula')

    F = f(20, 20)
    y = F.pdf(x[x > 0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x[x > 0], y)
    ax.set_title('F')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)
    
    st.divider()

    st.subheader("Proporciones")

    st.write("Intervalo de confianza para una proporción: ")

    st.latex(r"\hat{p}\pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")

    st.write("Intervalo de confianza para diferencia de proporciones: ")

    st.latex(r"\hat{p}_1 -\hat{p}_2\pm z_{\alpha/2}\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+\frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}")

    z = norm(0, 1)
    y = z.pdf(x)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.plot(x, y)
    ax.set_title('Z')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)
    
    st.code("=DISTR.NORM.ESTAND.INV(1 - alpha/2)", language='excelFormula')

with tab2:

    st.header("Elementos para una prueba de hipótesis")

    st.markdown("""
                - Hipótesis nula
                - Hipótesis alternativa
                - Estadístico de prueba
                - Región de rechazo
                """)
    

    x = np.linspace(-3, 3, 10000)
    T = t(30)
    y = T.pdf(x)
    fig, axes = plt.subplots(1, 3, figsize=(10,10))

    axes[0].plot(x, y)
    axes[1].plot(x, y)
    axes[2].plot(x, y)

    st.subheader("Prueba contra un valor")

    st.write("Prueba de hipótesis para medias")

    st.code("PRUEBA.T(col1;col2;1 o dos colas;par(1)hom(2)het(3))", language="excelFormula")

    st.latex(r"H_0: \mu = \mu_0")
    st.latex(r"H_1: \mu \neq \mu_0")
    st.latex(r"H_1: \mu > \mu_0")
    st.latex(r"H_1: \mu < \mu_0")

    st.write("Estadístico de prueba")

    st.latex(r"T=\frac{\bar{X}-\mu_0}{S/\sqrt{n}}")

    st.write("Región de rechazo")

    st.latex(r"t>t_\alpha \hspace{0.5cm}\text{cola superior}")
    st.latex(r"t<-t_\alpha \hspace{0.5cm}\text{cola inferior}")
    st.latex(r"|t|>t_{\alpha/2} \hspace{0.5cm}\text{dos colas}")

    st.subheader("Prueba entre dos muestras distintas e independientes")
    
    st.latex(r"H_0: \mu_1-\mu_2 = D_0")
    st.latex(r"H_1: \mu_1-\mu_2 > D_0")
    st.latex(r"H_1: \mu_1-\mu_2 < D_0")
    st.latex(r"H_1: \mu_1-\mu_2 \neq D_0")

    st.write("Estadístico de prueba")

    st.latex(r"T=\frac{\bar{x}_1-\bar{x}_2 - D_0}{S_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}")
    st.latex(r"S_p=\sqrt{\frac{(n_1-)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}}")

    st.write("Región de rechazo")

    st.latex(r"t>t_\alpha \hspace{0.5cm}\text{cola superior}")
    st.latex(r"t<-t_\alpha \hspace{0.5cm}\text{cola inferior}")
    st.latex(r"|t|>t_{\alpha/2} \hspace{0.5cm}\text{dos colas}")

 
