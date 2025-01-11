import streamlit as st
import time 
import os
from main import query_rag
from database import main as database, clear_database  # Importing the new function

DATA_PATH = "data"

# Turkish prompt template for querying financial documents
PROMPT_TEMPLATE = """
Aşağıdaki bağlamdan başka hiçbir bilgiye dayanmadan, soruya Türkçe dilinde doğru, kesin ve bağlama uygun bir cevap ver. Cevabında aşağıdaki kurallara uy:

1. Sorunun bağlamda açıkça yer alan cevabını ver. Eğer bağlamda yoksa "Bağlamda bu bilgi yer almıyor." şeklinde belirt.
2. Cevap mümkünse kısa ve öz olmalı, ancak gerektiğinde bağlamdaki ilgili detayları da içermelidir.
3. Cevap net bir şekilde yapılandırılmış olmalı ve gerektiğinde madde işaretleri veya numaralandırma kullanılmalıdır.
4. Cevabı bulduğun madde veya maddelerin numarasını açıkça belirt ve örneğin şu şekilde formatla: "(Kaynak: Madde X)".
5. Eğer bağlamda bulunan bilgiler birden fazla seçeneğe yol açıyorsa, en olası doğru seçeneği belirt ve kısa bir gerekçe ekle.
6. Her cevabın sonunda "Kesin ve net bilgi için bir vergi uzmanına danışılmalıdır." ifadesini ekle.

---

Bağlam:
{context}

---

Soru:
{question}

---

Cevap:
"""


# Streamlit UI
st.title("Finansal Döküman Tabanlı Soru-Cevap Sistemi")

# File manager
st.header("Adım 1: Veri Klasörünü Yönet")
if os.path.exists(DATA_PATH):
    files = os.listdir(DATA_PATH)
    if files:
        st.write("### Veri Klasöründeki Belgeler:")
        for file in files:
            file_path = os.path.join(DATA_PATH, file)
            col1, col2 = st.columns([3, 1])
            col1.write(file)  # Display file name
            # Add a delete button next to each file
            if col2.button("Sil", key=file):
                try:
                    os.remove(file_path)
                    st.success(f"{file} başarıyla silindi!")
                except Exception as e:
                    st.error(f"{file} silinemedi: {e}")
    else:
        st.info("Veri klasörü boş.")
else:
    st.warning(f"`{DATA_PATH}` klasörü mevcut değil.")

# File uploader for documents
st.header("Adım 2: Belgeleri Yükleyin")
uploaded_files = st.file_uploader(
    "PDF belgelerinizi yükleyin", type=["pdf"], accept_multiple_files=True
)

# Create two buttons side by side
col1, col2 = st.columns(2)

# "Yükle ve Veritabanını Güncelle" button
with col1:
    if st.button("Yükle ve Veritabanını Güncelle"):
        if uploaded_files:
            # Save uploaded files to DATA_PATH
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH)

            for uploaded_file in uploaded_files:
                with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.read())

            # Run the populate_database script
            st.info("Veritabanı Güncelleniyor...")
            database()
            st.success("Veritabanı başarıyla güncellendi!")
        else:
            st.warning("Lütfen en az bir belge yükleyin.")

# "Veritabanını Temizle" button
with col2:
    if st.button("Veritabanını Temizle"):
        try:
            clear_database()  # Call the function to clear the database
            st.success("Veritabanı başarıyla temizlendi!")
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")



# Model selector
st.header("Adım 3: Model Seçimi")
selected_model = st.selectbox(
    "Ollama modelini seçin:",
    options=["llama3.2:3b", "llama3.1:8b", "gemma2:2b", "gemma2:9b", "mistral:7b"],
    index=0
)

# Query input for asking questions
# Query input for asking questions
st.header("Adım 4: Soru")
query_text = st.text_input("Sorunuzu giriniz:")

if st.button("Çalıştır"):
    if query_text.strip():
        st.info("Veritabanı Sorgulanıyor...")

        # Start a spinner and timer
        start_time = time.time()
        with st.spinner("Soru işleniyor, lütfen bekleyin..."):
            response, sources = query_rag(query_text, PROMPT_TEMPLATE, selected_model)
        
        elapsed_time = time.time() - start_time
        elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time), 60)

        # Display the response
        st.success("Soru Yanıtlandı!")
        st.write("### Yanıt:")
        st.write(response)

        # Display elapsed time
        st.write(f"⏱ İşlem süresi: {elapsed_minutes} dakika, {elapsed_seconds} saniye")

        # Display the sources in expandable widgets
        st.write("### Kaynaklar:")
        if sources:
            for i, source in enumerate(sources):
                source_title = f"Kaynak {i + 1} - {source['source']})"
                with st.expander(source_title):
                    st.write(source["content"])
        else:
            st.warning("Kaynak bulunamadı.")
    else:
        st.warning("Lütfen bir soru girin.")

