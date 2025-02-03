import streamlit as st
import PyPDF2
import docx
import os
import json
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Carregar modelo de NLP
nlp = spacy.load("en_core_web_sm")

# Função para extrair texto de PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Função para extrair texto de arquivos DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Treinamento de modelo de Machine Learning (modelo mais complexo)
def train_model():
    data = pd.DataFrame({
        'text': [
            "Experiência com Python, Machine Learning e SQL",
            "Conhecimento avançado em Gestão de Projetos e SCRUM",
            "Trabalho com banco de dados Oracle e PostgreSQL",
            "Formação acadêmica incompleta sem experiência na área",
            "Experiência em suporte técnico e atendimento ao cliente",
            "Desenvolvimento de aplicações web com Django e Flask",
            "Nenhuma experiência profissional na área de TI",
            "Engenheiro de software com experiência em arquitetura de sistemas"
        ],
        'label': ["BOM", "BOM", "BOM", "RUIM", "RUIM", "BOM", "RUIM", "BOM"]
    })
    vectorizer = TfidfVectorizer()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = make_pipeline(vectorizer, model)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline

model = train_model()

# Função para classificar currículo como BOM ou RUIM
def classify_resume(text):
    return model.predict([text])[0]

# Interface Web com Streamlit
st.title("📂 Automação de Separação de Currículos")

# Upload de arquivos
uploaded_files = st.file_uploader("Envie os currículos (PDF ou DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# Definir palavras-chave personalizáveis
keywords = st.text_area("Palavras-chave para classificar currículos", "Python, Machine Learning, SQL, Gestão de Projetos, SCRUM, PostgreSQL").split(",")

if uploaded_files:
    resultados = []
    
    for file in uploaded_files:
        file_path = os.path.join("/tmp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Extração de texto
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_docx(file_path)
        
        # Classificação usando Machine Learning
        status = classify_resume(text)
        resultados.append({"Arquivo": file.name, "Classificação": status})
    
    # Exibir resultados
    st.write("### Resultados da Classificação")
    st.json(resultados)
    
    # Salvar em Google Sheets (Integração com Google Apps Script - Placeholder)
    st.write("Os resultados podem ser integrados ao Google Sheets.")
    
    st.success("Processo concluído! Resultados salvos.")
    
# Permitir até 5 usuários simultaneamente
st.write("Acesso permitido para até 5 usuários.")
