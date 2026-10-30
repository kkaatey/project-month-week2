from pathlib import Path

import streamlit as st

from search_engine import build_search_engine


st.set_page_config(
    page_title="Поиск товаров",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner="Готовим поисковый индекс")
def load_engine():
    return build_search_engine(Path("data"))


st.title("🔎 Поиск по каталогу товаров")
st.caption("Учебный MVP поиска на основе TF-IDF")

try:
    engine = load_engine()
except FileNotFoundError as error:
    st.error("Не найдены файлы данных")
    st.code(str(error))
    st.info("Поместите файлы Parquet в папку data и перезапустите приложение")
    st.stop()

with st.sidebar:
    st.header("Параметры")
    top_k = st.slider("Сколько товаров показать", 3, 20, 10)
    st.metric("Товаров в индексе", f"{len(engine.catalog):,}".replace(",", " "))
    st.caption("Модель учитывает название, категорию и описание")

query = st.text_input(
    "Поисковый запрос",
    placeholder="Например, белая блузка или джинсы на молнии",
)

if not query.strip():
    st.info("Введите запрос, чтобы увидеть результаты")
    st.stop()

results = engine.search(query, top_k)
st.subheader(f"Найдено: {len(results)}")

for position, row in enumerate(results.itertuples(index=False), 1):
    title = row.imt_name or "Товар без названия"
    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f"### {position}. {title.capitalize()}")
            details = " · ".join(
                value for value in [row.subj_name, row.subj_root_name, row.brand_name] if value
            )
            if details:
                st.caption(details)
            description = row.description or "Описание не указано"
            st.write(description[:500] + ("…" if len(description) > 500 else ""))
        with right:
            st.metric("Схожесть", f"{row.score:.3f}")
            st.caption(f"ID {row.imt_id}")
