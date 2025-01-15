import streamlit as st
import time
import os
from main import query_rag
from database import main as main, clear_database  # Importing the new function

DATA_PATH = "data"
CHROMA_PATHS = [
    "chroma/gelir_vergisi",
    "chroma/katma_deger",
    "chroma/ozel_tuketim",
    "chroma/kurumlar",
    "chroma/motorlu_tasit",
    ]  

# Turkish prompt template for querying financial documents
PROMPT_TEMPLATE = """
Aşağıdaki bağlamdan başka hiçbir bilgiye dayanmadan, soruya Türkçe dilinde doğru, kesin ve bağlama uygun bir cevap ver. Cevabında aşağıdaki kurallara uy:

1. Sorunun bağlamda açıkça yer alan cevabını ver. Eğer bağlamda yoksa "Bağlamda bu bilgi yer almıyor." şeklinde belirt.
2. Cevap net bir şekilde yapılandırılmış olmalı ve gerektiğinde madde işaretleri veya numaralandırma kullanılmalıdır.
3. Cevabı bulduğun madde veya maddelerin numarasını açıkça belirt ve örneğin şu şekilde formatla: "(Kaynak: Madde X)".
4. Eğer bağlamda bulunan bilgiler birden fazla seçeneğe yol açıyorsa, en olası doğru seçeneği belirt ve kısa bir gerekçe ekle.
5. Her cevabın sonunda "Kesin ve net bilgi için bir vergi uzmanına danışılmalıdır." ifadesini ekle.

---

Bağlam:
{context}

---

Soru:
{question}

---

Cevap:
"""

def category_chooser(category):
    """
    Converts the selected category to an integer for the main function.
    """
    if category == "gelir vergisi":
        selected = 0
    elif category == "katma değer vergisi":
        selected = 1
    elif category == "özel tüketim vergisi":
        selected = 2
    elif category == "kurumlar vergisi":
        selected = 3
    elif category == "motorlu taşıtlar vergisi":
        selected = 4

    return selected    


# Streamlit UI
st.title("Finansal Döküman Tabanlı Soru-Cevap Sistemi")

# Step 1: Content Category Selection
st.header("Adım 1: Soru İçeriği hangi Kanun Maddelerini İlgilendiriyor?")
category = st.selectbox(
    "Soru içeriği hangi kategoriye ait?",
    [
        "gelir vergisi",
        "katma değer vergisi",
        "özel tüketim vergisi",
        "kurumlar vergisi",
        "motorlu taşıtlar vergisi",
    ],
)

if category:
    st.success(f"Seçilen kategori: {category}")

    # Step 2: File manager
    st.header("Adım 2: Veri Klasörünü Yönet")
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

    # Step 3: File uploader for documents
    st.header("Adım 3: Belgeleri Yükleyin")
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
                tax_category = CHROMA_PATHS[category_chooser(category)]
                print(f"Tax Category: {tax_category}")
                main(tax_category)
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

    # Step 4: Model selector
    st.header("Adım 4: Model Seçimi")
    selected_model = st.selectbox(
        "Ollama modelini seçin:",
        options=["gemma2:2b", "gemma2:9b", "mistral:7b"],
        index=0,
    )

    # Step 5: Query input for asking questions
    st.header("Adım 5: Soru")
    query_text = st.text_input("Sorunuzu giriniz:")

    if st.button("Çalıştır"):
        if query_text.strip():
            st.info("Veritabanı Sorgulanıyor...")

            # Start a spinner and timer
            start_time = time.time()
            tax_category = CHROMA_PATHS[category_chooser(category)]
            print(f"Tax Category 2: {tax_category}")
            with st.spinner("Soru işleniyor, lütfen bekleyin..."):
                response, sources = query_rag(query_text, PROMPT_TEMPLATE, selected_model, tax_category)
            
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
                    source_title = f"Kaynak {i + 1} - ({source['source']} - Confidence Score: {source['score']:.2f})"
                    with st.expander(source_title):
                        st.write(source["content"])
            else:
                st.warning("Kaynak bulunamadı.")
        else:
            st.warning("Lütfen bir soru girin.")
