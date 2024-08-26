import streamlit as st
#creamos un widget para subir un archivo del mejor modelo entrenado
uploaded_file=st.file_uploader("Sube evaluate_model",type=["txt"])
#como verificamos si se ha subido el archivo 
if uploaded_file is not None:
    file_content=uploaded_file.read()
    st.text("Contenido del archivo:")
    st.write(file_content)
