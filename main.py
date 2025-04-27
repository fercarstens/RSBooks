import streamlit as st
import pandas as pd
import numpy as np
import glob
from sklearn.metrics.pairwise import cosine_similarity
import json

# Configuración
st.set_page_config(page_title="Recomendador de Libros", layout="centered")
st.title("📖 Recomendador de Libros con IA")
st.markdown("Ingresa de 1 a 3 libros que te hayan gustado y te recomendaremos otros similares.")


# Cargar dataset preprocesado
@st.cache_data
def load_books():
    parquet_files = glob.glob("data/books_part_*.parquet") 
    if not parquet_files:
        st.error("No se encontraron archivos .parquet.")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.dropna(subset=["title", "description", "embedding"])
    return df

df = load_books()
random_title = df["title"].sample(1).values[0]
if df.empty:
    st.warning("El dataset está vacío o no fue cargado correctamente.")
else:
    st.success(f'Dataset cargado correctamente con {df.shape[0]} libros. Ejemplo: "{random_title}"')


# Normalización única al cargar el DataFrame
df["title_norm"] = df["title"].str.strip().str.lower()

# Formulario de libros favoritos
with st.form("form_recommend"):
    titles_input = st.text_area("Libros favoritos (uno por línea)", value="", height=100,
                                 placeholder="Ejemplo (en inglés): Twilight\nThe Notebook\nDivergent")
    st.caption("⚠️ Los títulos deben estar escritos en inglés para que puedan ser reconocidos.")
    submitted = st.form_submit_button("Recomendar 🔍")

if submitted:
    favorite_titles = [t.strip().lower() for t in titles_input.splitlines() if t.strip()]

    if not favorite_titles:
        st.warning("Por favor, escribe al menos un libro.")
    else:
        matched = df[df['title_norm'].isin(favorite_titles)]

        if matched.empty:
            st.error("Ninguno de los títulos fue reconocido. Recuerda escribirlos en inglés y revisa la ortografía.")
        else:
            st.success(f"{len(matched)} libro(s) encontrado(s). Calculando recomendaciones...")

            user_embedding = np.mean(matched['embedding'].tolist(), axis=0).reshape(1, -1)

            df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], user_embedding)[0][0])

            df_filtered = df[~df['title_norm'].isin(favorite_titles)]

            top_recommendations = df_filtered.sort_values(by='similarity', ascending=False).head(3)

            st.subheader("📚 Recomendaciones:")
            for _, row in top_recommendations.iterrows():
                st.markdown(f"**{row['title']}** — *{row['author']}*")
                st.markdown(f"Descripción: {row['description']}")
                st.markdown(f"_Género:_ {row.get('genres', 'Desconocido')}")
                st.markdown(f"🔁 Similitud: `{row['similarity']:.2f}`")
                st.markdown("---")
