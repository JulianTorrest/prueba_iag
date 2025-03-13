import streamlit as st
import PyPDF2
from docx import Document
import pandas as pd
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from nltk.corpus import stopwords
import speech_recognition as sr
from gtts import gTTS
import os
from langdetect import detect
from PIL import Image
import pytesseract
from youtube_transcript_api import YouTubeTranscriptApi
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    print("Recurso Punkt descargado correctamente.")
except LookupError:
    print("Descargando recurso Punkt...")
    nltk.download('punkt')
    print("Recurso Punkt descargado.")

try:
    nltk.data.find('tokenizers/punkt_tab/spanish')
    print("Recurso Punkt_tab para español descargado correctamente.")
except LookupError:
    print("Descargando recurso Punkt_tab para español...")
    nltk.download('punkt_tab')
    print("Recurso Punkt_tab para español descargado.")

try:
    stopwords.words('english')  # Intenta cargar stopwords para verificar la descarga
    print("Corpus de stopwords descargado correctamente.")
except LookupError:
    print("Descargando corpus de stopwords...")
    nltk.download('stopwords')
    print("Corpus de stopwords descargado.")


# Funciones de Lectura de Archivos
def leer_pdf(archivo):  # Modifica para aceptar UploadedFile
    try:
        lector_pdf = PyPDF2.PdfReader(archivo)  # Usa el objeto UploadedFile directamente
        texto = ""
        for pagina in lector_pdf.pages:
            texto += pagina.extract_text()
        return texto
    except Exception as e:
        return f"Error al leer PDF: {e}"
        
def leer_word(archivo):
    try:
        documento = Document(archivo)
        texto = ""
        for parrafo in documento.paragraphs:
            texto += parrafo.text + "\n"
        return texto
    except Exception as e:
        return f"Error al leer Word: {e}"

def leer_csv(archivo):
    try:
        df = pd.read_csv(archivo)
        texto = df.to_string()
        return texto
    except Exception as e:
        return f"Error al leer CSV: {e}"

def leer_web(url):
    try:
        respuesta = requests.get(url)
        soup = BeautifulSoup(respuesta.content, 'html.parser')
        texto = soup.get_text()
        return texto
    except Exception as e:
        return f"Error al leer la página web: {e}"

def leer_web_idioma(url):
    try:
        texto = leer_web(url)
        if texto:
            idioma = detect(texto)
            return texto, idioma
        else:
            return "No se pudo acceder a la página web.", None
    except Exception as e:
        return f"Error al leer la página web: {e}", None

def leer_imagen(archivo):
    try:
        imagen = Image.open(archivo)
        texto = pytesseract.image_to_string(imagen, lang="spanish")  # Cambia el idioma si es necesario
        return texto
    except Exception as e:
        return f"Error al leer imagen: {e}"

def extraer_transcripcion_youtube(url):
    try:
        video_id = url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['spanish'])  # Cambia el idioma si es necesario
        texto = " ".join([entry['text'] for entry in transcript])
        return texto
    except Exception as e:
        return f"Error al extraer transcripción de YouTube: {e}"

# Funciones de Preprocesamiento de Texto
def preprocesar_texto(texto):
    try:
        tokens = word_tokenize(texto.lower(), language='spanish')
        print(f"Tokens después de word_tokenize: {tokens}")
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        print(f"Tokens después de lematización: {tokens}")
        stop_words = set(stopwords.words('spanish'))
        tokens = [token for token in tokens if token not in stop_words]
        print(f"Tokens después de stopwords: {tokens}")
        return tokens
    except Exception as e:
        print(f"Error en preprocesar_texto: {e}")
        return []

def reconocer_entidades(texto):
    doc = nlp(texto)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    return entidades


def analizar_sentimientos(texto):
    analizador = SentimentIntensityAnalyzer()
    puntuaciones = analizador.polarity_scores(texto)
    if puntuaciones["compound"] >= 0.05:
        return "positivo", puntuaciones["compound"]
    elif puntuaciones["compound"] <= -0.05:
        return "negativo", puntuaciones["compound"]
    else:
        return "neutro", puntuaciones["compound"]

def resaltar_informacion(texto):
    sentimiento = analizar_sentimientos(texto)
    entidades = reconocer_entidades(texto)
    texto_resaltado = texto
    if sentimiento == "positivo":
        texto_resaltado = f'<span style="background-color: lightgreen;">{texto_resaltado}</span>'
    elif sentimiento == "negativo":
        texto_resaltado = f'<span style="background-color: lightcoral;">{texto_resaltado}</span>'
    for entidad, tipo in entidades:
        texto_resaltado = texto_resaltado.replace(entidad, f'<span style="background-color: lightyellow;">{entidad} ({tipo})</span>')
    return texto_resaltado


palabras_positivas = ["feliz", "bien", "excelente", "amor", "alegría"]  # Ejemplo
palabras_negativas = ["triste", "mal", "pésimo", "odio", "dolor"]  # Ejemplo

def generar_nube_palabras_sentimiento(texto, palabras_sentimiento, titulo, idioma="es"):
    tokens = preprocesar_texto(texto)
    stop_words = set(stopwords.words(idioma))
    palabras_filtradas = [token for token in tokens if token in palabras_sentimiento and token not in stop_words]
    texto_limpio = " ".join(palabras_filtradas)
    if texto_limpio:
        wordcloud = WordCloud().generate(texto_limpio)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(titulo)
        st.pyplot(plt)

# Funciones de Búsqueda Mejoradas
def buscar_coincidencia_parcial(texto, consulta):
    tokens_texto = preprocesar_texto(texto)
    tokens_consulta = preprocesar_texto(consulta)
    resultados = [token for token in tokens_texto if any(consulta_token in token for consulta_token in tokens_consulta)]
    return resultados

def contar_frecuencia_palabras(texto, consulta):
    tokens_texto = preprocesar_texto(texto)
    tokens_consulta = preprocesar_texto(consulta)
    frecuencia = sum(1 for token in tokens_texto if token in tokens_consulta)
    return frecuencia

# Funciones de Resumen de Texto (Básicas)
def resumir_texto(texto, num_oraciones=3):
    try:
        oraciones = sent_tokenize(texto, language='spanish') #Se agrega el language
        tokens = preprocesar_texto(texto)
        frecuencia_palabras = nltk.FreqDist(tokens)
        oraciones_importantes = sorted(oraciones, key=lambda oracion: sum(frecuencia_palabras[token] for token in word_tokenize(oracion.lower()) if token.isalnum()), reverse=True)
        return ' '.join(oraciones_importantes[:num_oraciones])
    except Exception as e:
        print(f"Error en resumir_texto: {e}")
        return "Error al resumir el texto."


# Resúmenes de Múltiples Fuentes con Citas
def resumir_multiples_fuentes(fuentes, num_oraciones=3):
    resumenes = []
    citas = []
    for fuente in fuentes:
        texto = ""
        if fuente.endswith(".pdf"):
            texto = leer_pdf(fuente)
        elif fuente.endswith(".docx"):
            texto = leer_word(fuente)
        elif fuente.endswith(".csv"):
            texto = leer_csv(fuente)
        elif fuente.startswith("http://") or fuente.startswith("https://"):
            texto = leer_web(fuente)
        elif fuente.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
          texto = leer_imagen(fuente)
        if texto:
            resumen = resumir_texto(texto, num_oraciones)
            resumenes.append(resumen)
            citas.append(f"Resumen de: {fuente}")
    return "\n".join(resumenes), "\n".join(citas)


# Consultas en Páginas Web Específicas
def consultar_pagina_web(url, consulta):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
        soup = BeautifulSoup(response.content, 'html.parser')
        texto_pagina = soup.get_text()
        print(f"Texto extraído de la página web: {texto_pagina}")  # Imprime el texto extraído
        resultados = [linea for linea in texto_pagina.splitlines() if consulta.lower() in linea.lower()]
        print(f"Resultados de la búsqueda: {resultados}")  # Imprime los resultados
        return resultados
    except requests.exceptions.RequestException as e:
        print(f"Error al acceder a la página web: {e}")
        return []
    except Exception as e:
        print(f"Error en consultar_pagina_web: {e}")
        return []


# Generación de Gráficos y Nubes de Palabras
def generar_grafico_frecuencia(texto):
    tokens = preprocesar_texto(texto)
    frecuencia = nltk.FreqDist(tokens)
    frecuencia.plot(20)
    st.pyplot(plt)

def generar_nube_palabras(texto, idioma):
    try:
        print(f"Idioma seleccionado: {idioma}")  # Imprime el idioma
        stop_words = set(stopwords.words(idioma))
        tokens = word_tokenize(texto.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        texto_limpio = ' '.join(tokens)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_limpio)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error al generar la nube de palabras: {e}")

def generar_grafico_barras(texto, num_palabras=10):
    tokens = preprocesar_texto(texto)
    frecuencia = nltk.FreqDist(tokens)
    palabras = [palabra for palabra, _ in frecuencia.most_common(num_palabras)]
    conteo = [frecuencia[palabra] for palabra in palabras]
    df = pd.DataFrame({'Palabra': palabras, 'Conteo': conteo})
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Conteo', y='Palabra', data=df)
    plt.title('Palabras Clave Más Frecuentes')
    plt.xlabel('Conteo')
    plt.ylabel('Palabra')
    st.pyplot(plt)

def generar_grafico_entidades(texto):
    entidades = reconocer_entidades(texto)
    tipos_entidades = {}
    for _, tipo in entidades:
        tipos_entidades[tipo] = tipos_entidades.get(tipo, 0) + 1
    if tipos_entidades:
        df = pd.DataFrame({'Entidad': list(tipos_entidades.keys()), 'Conteo': list(tipos_entidades.values())})
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Conteo', y='Entidad', data=df)
        plt.title('Entidades Reconocidas')
        plt.xlabel('Conteo')
        plt.ylabel('Entidad')
        st.pyplot(plt)


# Descarga de Resúmenes
def descargar_resumen(resumen, nombre_archivo="resumen.txt"):
    b64 = base64.b64encode(resumen.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{nombre_archivo}">Descargar Resumen</a>'
    return href

# Resúmenes por Voz
def grabar_voz():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Habla ahora...")
        audio = recognizer.listen(source)
    try:
        texto = recognizer.recognize_google(audio, language="spanish")  # Cambia el idioma si es necesario
        return texto
    except sr.UnknownValueError:
        return "No se pudo entender el audio."
    except sr.RequestError:
        return "Error en la solicitud de reconocimiento de voz."

def texto_a_voz(texto, idioma="spanish"):
    tts = gTTS(texto, lang=idioma)
    tts.save("resumen_voz.mp3")
    audio_file = open("resumen_voz.mp3", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    os.remove("resumen_voz.mp3")

# Interfaz de Streamlit
st.title("IA de Búsqueda de Documentos Avanzada")

st.sidebar.title("Opciones Globales")
idioma_interfaz = st.sidebar.selectbox("Idioma de la interfaz:", ["spanish", "english"])
tema = st.sidebar.selectbox("Tema:", ["Claro", "Oscuro"])

with st.expander("Búsqueda y Resumen de Documentos"):
    consulta = st.text_input("Ingresa tu consulta:")
    archivo = st.file_uploader("Carga un archivo (PDF, Word, CSV, imagen):")
    url_youtube = st.text_input("Ingresa un enlace de YouTube:")
    idioma_archivo = st.selectbox("Idioma del archivo:", ["spanish", "english"])
    idioma_resumen = st.selectbox("Idioma del resumen:", ["spanish", "english"])
    if st.button("Buscar"):
        # Lógica de búsqueda y resumen
        if archivo is not None:
            if archivo.name.endswith(".pdf"):
                st.session_state['texto'] = leer_pdf(archivo)
            elif archivo.name.endswith(".docx"):
                st.session_state['texto'] = leer_word(archivo)
            elif archivo.name.endswith(".csv"):
                st.session_state['texto'] = leer_csv(archivo)
            elif archivo.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                st.session_state['texto'] = leer_imagen(archivo)
            else:
                st.write("Formato de archivo no compatible.")
                st.session_state['texto'] = ""
        elif url_youtube:
            st.session_state['texto'] = extraer_transcripcion_youtube(url_youtube)
        else:
            st.session_state['texto'] = ""

        if st.session_state.get('texto') and st.session_state.get('texto').strip():
            with st.spinner("Procesando..."):
                resumen = resumir_texto(st.session_state.get('texto'))
                coincidencias = buscar_coincidencia_parcial(st.session_state.get('texto'), consulta)
                frecuencia = contar_frecuencia_palabras(st.session_state.get('texto'), consulta)
                tab1, tab2, tab3 = st.tabs(["Resumen", "Coincidencias", "Frecuencia"])
                with tab1:
                    st.markdown("### Resumen:")
                    st.write(resumen)
                with tab2:
                    st.markdown("### Coincidencias:")
                    st.write(coincidencias)
                with tab3:
                    st.markdown("### Frecuencia:")
                    st.write(frecuencia)
        else:
            st.write("No se ha proporcionado texto para buscar.")

with st.expander("Resumen de Múltiples Fuentes"):
    fuentes_multiples = st.text_area("URLs o rutas de archivos (separadas por comas):")
    if st.button("Resumen Múltiple"):
        fuentes = [f.strip() for f in fuentes_multiples.split(",")]
        resumen, citas = resumir_multiples_fuentes(fuentes)
        st.markdown("### Resumen:")
        st.write(resumen)
        st.markdown("### Citas:")
        st.write(citas)

with st.expander("Consulta en Página Web"):
    url_consulta = st.text_input("URL para consulta específica:")
    if st.button("Consultar Página Web"):
        resultados = consultar_pagina_web(url_consulta, consulta)
        st.markdown("### Resultados de la consulta:")
        st.write(resultados)

with st.expander("Resumen por Texto Escrito"):
    if st.session_state.get('texto'):  # Usa el texto cargado del archivo
        resumen = resumir_texto(st.session_state.get('texto'))
        st.markdown("### Resumen del texto del archivo:")
        st.write(resumen)
        st.markdown(descargar_resumen(resumen, "resumen_archivo.txt"), unsafe_allow_html=True)
    else:
        st.write("No hay texto disponible para resumir. Carga un archivo o ingresa texto en la sección 'Búsqueda y Resumen de Documentos'.")

with st.expander("Resumen por Voz"):
    if st.button("Grabar Voz y Resumir"):
        texto_voz = grabar_voz()
        if texto_voz:
            st.write(f"Texto grabado: {texto_voz}")
            resumen_voz = resumir_texto(texto_voz)
            st.write(f"Resumen: {resumen_voz}")
            texto_a_voz(resumen_voz, idioma_resumen)

with st.expander("Generación de Gráficos"):
    if st.button("Generar Gráfico de Frecuencia"):
        if st.session_state.get('texto'):
            generar_grafico_frecuencia(st.session_state.get('texto'))
    if st.button("Generar Nube de Palabras"):
        if st.session_state.get('texto'):
            generar_nube_palabras(st.session_state.get('texto'), idioma_archivo)
    if st.button("Generar Grafico de Barras"):
        if st.session_state.get('texto'):
            generar_grafico_barras(st.session_state.get('texto'), num_palabras=10)
    if st.button("Generar Nube de Palabras Positivas"):
        if st.session_state.get('texto'):
            generar_nube_palabras_sentimiento(st.session_state.get('texto'), palabras_positivas, "Palabras Positivas", idioma_archivo)
    if st.button("Generar Nube de Palabras Negativas"):
        if st.session_state.get('texto'):
            generar_nube_palabras_sentimiento(st.session_state.get('texto'), palabras_negativas, "Palabras Negativas", idioma_archivo)
    if st.button("Generar Gráfico de Entidades"):
        if st.session_state.get('texto'):
            generar_grafico_entidades(st.session_state.get('texto'))

#st.help("Carga un archivo o ingresa una URL para buscar y resumir información.")
