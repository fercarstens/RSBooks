
# 📚 Recomendador de Libros con IA

Este proyecto implementa un sistema inteligente de recomendación de libros basado en el contenido de las descripciones. Los usuarios ingresan entre 1 y 3 títulos de libros que les han gustado, y la app les sugiere otros libros similares utilizando modelos de procesamiento del lenguaje natural (PLN).

---

## 🚀 Tecnologías utilizadas

- Python 3.10+
- [Streamlit](https://streamlit.io/) — para la interfaz web
- [pandas](https://pandas.pydata.org/) — para manipulación de datos
- [SentenceTransformers](https://www.sbert.net/) — para generar embeddings semánticos
- [scikit-learn](https://scikit-learn.org/stable/) — para calcular similitudes
- NLTK — para procesamiento de texto
- Parquet — formato eficiente para datos tabulares

---

## 🧠 ¿Cómo funciona?

1. El sistema recibe una lista de libros favoritos del usuario.
2. Busca esos títulos dentro del dataset.
3. Calcula un vector promedio usando embeddings precomputados de las descripciones de los libros.
4. Compara ese vector con el resto del dataset usando **cosine similarity**.
5. Devuelve los libros más similares, excluyendo los ya seleccionados.

---

## 📂 Estructura del proyecto

```
📦 book-recommender
├── 📁 data/                 # Archivos .parquet con embeddings
│   ├── books_part_1.parquet
│   ├── books_part_2.parquet
│   └── ...
├── main.app                # Aplicación Streamlit
└── requirements.txt        # Dependencias del proyecto
```

---

## ▶️ Cómo ejecutar la app

1. Clona este repositorio:

```bash
git clone https://github.com/fercarstens/rsbooks.git
cd rsbooks
```

2. Crea un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecuta la app:

```bash
streamlit run main.app
```

---

## 📊 Dataset

- Fuente: Compilado con dos Datasets con metadatos de libros (autores, géneros y descripciones) descargados de kaggle:
    https://www.kaggle.com/datasets/meetnaren/goodreads-best-books
    https://www.kaggle.com/datasets/elvinrustam/books-dataset
- Preprocesado y dividido en archivos `.parquet` para mejorar el rendimiento.
- Embeddings generados con el modelo `all-MiniLM-L6-v2`.

---

## 📄 Licencia

Este proyecto es parte de un trabajo académico para el Máster en Inteligencia Artificial del CEI Sevilla. Uso libre para fines educativos y no comerciales.

---
