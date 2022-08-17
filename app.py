from processing import TextProcessing
import streamlit as st
import pickle

model = pickle.load(open('modelos_salvos/finalized_model.sav', 'rb'))

st.markdown("#### Modelo de classificação de música Heavy Metal")

uploaded_file = st.file_uploader("Escolha o arquivo de texto com a música")
if uploaded_file:

    lyrics = uploaded_file.read().decode('utf-8')
    st.write("Arquivo:", uploaded_file.name)

    processed_text = TextProcessing(lyrics)
    x = processed_text.format_text()
    predict = round(model.predict_proba(x).max(), 2)*100

    if model.predict(x) == 1:
        st.success(f"É Heavy Metal :guitar: com precisão de {predict}%")
    else:
        st.error(f"Não é Heavy Metal :sleeping: com precisão de {predict}%")