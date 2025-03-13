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
def preprocesar_texto(texto, idioma="spanish"):
    """
    Preprocesa un texto realizando tokenización, lematización y eliminación de stopwords.
    Args:
        texto (str): El texto a preprocesar.
        idioma (str, optional): Idioma de las stopwords. Defaults to "spanish".
    Returns:
        list: Lista de tokens preprocesados.
    """
    try:
        tokens = word_tokenize(texto.lower(), language=idioma)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        stop_words = set(stopwords.words(idioma))
        tokens = [token for token in tokens if token not in stop_words]
        return tokens
    except Exception as e:
        print(f"Error en preprocesar_texto: {e}")
        return []

def buscar_coincidencia_parcial(texto, consulta, idioma="spanish"):
    """
    Busca coincidencias parciales de palabras entre un texto y una consulta.
    Args:
        texto (str): El texto donde buscar.
        consulta (str): La consulta a buscar.
        idioma (str, optional): Idioma para el preprocesamiento. Defaults to "spanish".
    Returns:
        list: Lista de palabras del texto que coinciden parcialmente con la consulta.
    """
    try:
        tokens_texto = preprocesar_texto(texto, idioma)
        tokens_consulta = preprocesar_texto(consulta, idioma)
        # Usar conjuntos para una búsqueda más eficiente
        conjunto_consulta = set(tokens_consulta)
        resultados = [token for token in tokens_texto if any(consulta_token in token for consulta_token in conjunto_consulta)]
        return resultados
    except Exception as e:
        print(f"Error en buscar_coincidencia_parcial: {e}")
        return []

def contar_frecuencia_palabras(texto, consulta, idioma="spanish"):
    """
    Cuenta la frecuencia de palabras de la consulta en el texto.
    Args:
        texto (str): El texto donde contar.
        consulta (str): La consulta a buscar.
        idioma (str, optional): Idioma para el preprocesamiento. Defaults to "spanish".
    Returns:
        int: La frecuencia de palabras de la consulta en el texto.
    """
    try:
        tokens_texto = preprocesar_texto(texto, idioma)
        tokens_consulta = preprocesar_texto(consulta, idioma)
        # Usar conjuntos para una búsqueda más eficiente
        conjunto_consulta = set(tokens_consulta)
        frecuencia = sum(1 for token in tokens_texto if token in conjunto_consulta)
        return frecuencia
    except Exception as e:
        print(f"Error en contar_frecuencia_palabras: {e}")
        return 0
        
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

def generar_nube_palabras_sentimiento(texto, palabras_sentimiento, titulo, idioma="spanish"):
    """
    Genera una nube de palabras basada en el sentimiento del texto.

    Args:
        texto (str): El texto para generar la nube de palabras.
        palabras_sentimiento (list): Lista de palabras relacionadas con el sentimiento (positivo o negativo).
        titulo (str): Título para la nube de palabras.
        idioma (str, optional): Idioma de las stopwords. Defaults to "spanish".
    """
    try:
        # Preprocesar el texto
        tokens = word_tokenize(texto.lower())
        stop_words = set(stopwords.words(idioma))
        palabras_filtradas = [token for token in tokens if token in palabras_sentimiento and token.isalpha() and token not in stop_words]
        texto_limpio = " ".join(palabras_filtradas)

        # Imprimir información de depuración
        print(f"Texto limpio para {titulo}: {texto_limpio}")
        print(f"Palabras filtradas para {titulo}: {palabras_filtradas}") # Imprimir palabras filtradas

        # Generar la nube de palabras si hay palabras filtradas
        if texto_limpio:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_limpio)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(titulo)
            st.pyplot(plt)
        else:
            st.warning(f"No se encontraron palabras para generar la nube de palabras {titulo}.")

    except Exception as e:
        st.error(f"Error al generar la nube de palabras {titulo}: {e}")

# Funciones de Resumen de Texto (Básicas)
def resumir_texto(texto, limite_palabras=1000, idioma="spanish"):
    """
    Genera un resumen de un texto con un límite de palabras.
    Args:
        texto (str): El texto a resumir.
        limite_palabras (int, optional): Número aproximado de palabras para el resumen. Defaults to 500.
        idioma (str, optional): Idioma del texto. Defaults to "spanish".

    Returns:
        str: Resumen del texto.
    """
    try:
        parser = PlaintextParser.from_string(texto, Tokenizer(idioma))
        stemmer = Stemmer(idioma)
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(idioma)
        sentences = summarizer(parser.document, sentences_count=10) # Ajusta sentences_count según la longitud del texto
        resumen_completo = " ".join([str(sentence) for sentence in sentences])
        palabras_resumen = resumen_completo.split()
        if len(palabras_resumen) > limite_palabras:
            resumen_final = " ".join(palabras_resumen[:limite_palabras])
        else:
            resumen_final = resumen_completo
        return resumen_final
    except Exception as e:
        print(f"Error en resumir_texto: {e}")
        return "No se pudo generar el resumen."
        
# Resúmenes de Múltiples Fuentes con Citas
def resumir_multiples_fuentes(texto_combinado, idioma_resumen="spanish", limite_palabras=500):
    if isinstance(texto_combinado, str): #verifica que la variable sea de tipo string
        resumen = resumir_texto(texto_combinado, idioma=idioma_resumen, limite_palabras=limite_palabras)
        return resumen
    else:
        return "No se ha proporcionado texto válido para resumir."
        
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

# Funciones de Búsqueda Mejoradas (Nuevas funciones de búsqueda)
def buscar_coincidencia_exacta(texto, consulta):
    try:
        resultados = [linea for linea in texto.splitlines() if consulta.lower() in linea.lower()]
        return resultados
    except Exception as e:
        print(f"Error en buscar_coincidencia_exacta: {e}")
        return []

def contar_frecuencia_exacta(texto, consulta):
    try:
        frecuencia = texto.lower().count(consulta.lower())
        return frecuencia
    except Exception as e:
        print(f"Error en contar_frecuencia_exacta: {e}")
        return 0

def buscar_coincidencia_regex(texto, consulta):
    try:
        resultados = re.findall(consulta, texto, re.IGNORECASE)
        return resultados
    except Exception as e:
        print(f"Error en buscar_coincidencia_regex: {e}")
        return []
        
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
                st.markdown("### Resumen:")
                st.write(resumen)

                if consulta:
                    # Búsqueda directa
                    coincidencias_exactas = buscar_coincidencia_exacta(st.session_state.get('texto'), consulta)
                    frecuencia_exacta = contar_frecuencia_exacta(st.session_state.get('texto'), consulta)

                    # Búsqueda con regex
                    coincidencias_regex = buscar_coincidencia_regex(st.session_state.get('texto'), consulta)

                    st.markdown("### Resultados de la Búsqueda:")
                    st.markdown("#### Coincidencias Exactas:")
                    st.write(coincidencias_exactas)
                    st.markdown("#### Frecuencia Exacta:")
                    st.write(frecuencia_exacta)
                    st.markdown("#### Coincidencias Regex:")
                    st.write(coincidencias_regex)
                else:
                    st.write("Ingresa una consulta para buscar.")
        else:
            st.write("No se ha proporcionado texto para buscar.")

with st.expander("Resumen de Múltiples Fuentes"):
    idioma_resumen_multiple = st.selectbox("Idioma del resumen:", ["spanish", "english"], key="idioma_resumen_multiple")
    if st.button("Generar Resumen Múltiple"):
        if st.session_state.get('texto') and st.session_state.get('texto').strip():
            texto_combinado = st.session_state.get('texto')
            if isinstance(texto_combinado, str): #verifica que la variable sea de tipo string
                resumen_multiple = resumir_multiples_fuentes(texto_combinado, idioma_resumen_multiple)
                st.markdown("### Resumen de Múltiples Fuentes:")
                st.write(resumen_multiple)
            else:
                st.write("El texto cargado no es válido.")
        else:
            st.write("No se han cargado documentos o URLs para resumir.")

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

#st.help("Carga un archivo o ingresa una URL para buscar y resumir información.")
