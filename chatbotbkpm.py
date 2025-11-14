import streamlit as st
from dotenv import load_dotenv
import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO

# Load environment variables
load_dotenv()

# --- Konfigurasi Awal Streamlit ---
st.set_page_config(page_title="Chatbot Tanya Jawab dengan Pinecone & OpenAI", page_icon="💬")
st.title("💬 Chatbot Tanya Jawab")
st.markdown("Unggah dokumen Anda (PDF atau TXT) untuk membuat basis pengetahuan, lalu ajukan pertanyaan kepada chatbot.")

# --- Sidebar untuk API Keys dan Konfigurasi ---
with st.sidebar:
    st.header("Konfigurasi")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Dapatkan dari platform.openai.com")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", help="Dapatkan dari app.pinecone.io")
    pinecone_environment = st.text_input("Pinecone Environment", help="Contoh: us-west-2")
    pinecone_index_name = st.text_input("Nama Indeks Pinecone", "my-rag-index", help="Nama indeks Pinecone yang akan digunakan")

    if not (openai_api_key and pinecone_api_key and pinecone_environment and pinecone_index_name):
        st.warning("Mohon masukkan semua API Key dan konfigurasi Pinecone di sidebar.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.subheader("Unggah Dokumen")
    uploaded_file = st.file_uploader("Pilih file PDF atau TXT", type=["pdf", "txt"])
    process_button = st.button("Proses Dokumen")

# --- Inisialisasi Pinecone dan OpenAI ---
@st.cache_resource
def initialize_pinecone_and_openai(pinecone_api_key, pinecone_environment, pinecone_index_name):
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") # Dimenasi 1536
        
        # Cek apakah indeks ada, jika tidak, buat
        if pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(name=pinecone_index_name, dimension=1536, metric="cosine")
            st.info(f"Indeks Pinecone '{pinecone_index_name}' berhasil dibuat.")
        
        vectorstore = Pinecone(index_name=pinecone_index_name, embedding=embeddings)
        return embeddings, vectorstore
    except Exception as e:
        st.error(f"Gagal menginisialisasi Pinecone atau OpenAI Embeddings: {e}")
        return None, None

embeddings, vectorstore = initialize_pinecone_and_openai(pinecone_api_key, pinecone_environment, pinecone_index_name)

if not embeddings or not vectorstore:
    st.stop()

# --- Fungsi untuk Memproses Dokumen ---
def process_document(uploaded_file, embeddings, vectorstore):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        temp_file_path = f"./temp_uploaded_file.{file_extension}"

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = None
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == "txt":
            loader = TextLoader(temp_file_path)
        else:
            st.error("Format file tidak didukung. Mohon unggah file PDF atau TXT.")
            return

        if loader:
            with st.spinner("Membaca dan memproses dokumen..."):
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                # Mengunggah ke Pinecone secara batch
                batch_size = 100
                total_batches = (len(texts) + batch_size - 1) // batch_size
                progress_bar = st.progress(0)

                for i in range(total_batches):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, len(texts))
                    batch = texts[start_index:end_index]
                    
                    # Konversi Document objek menjadi string untuk embedding
                    # dan kemudian menjadi tuple (id, embedding, metadata) untuk upsert langsung
                    # atau menggunakan vectorstore.add_documents() jika itu mendukung batching
                    
                    # Untuk kesederhanaan, kita akan menggunakan add_documents, yang menangani embedding secara internal
                    # namun perlu diingat bahwa ini akan memanggil API embedding untuk setiap dokumen dalam batch.
                    # Jika Anda perlu mengoptimalkan, Anda bisa mendapatkan embedding secara terpisah dan kemudian upsert langsung.
                    
                    vectorstore.add_documents(batch)
                    progress = (i + 1) / total_batches
                    progress_bar.progress(progress)
                
                st.success(f"Dokumen '{uploaded_file.name}' berhasil diunggah ke Pinecone.")
            os.remove(temp_file_path)

# --- Proses Dokumen saat tombol ditekan ---
if process_button:
    if uploaded_file:
        process_document(uploaded_file, embeddings, vectorstore)
    else:
        st.warning("Mohon unggah file terlebih dahulu.")

# --- Inisialisasi Model Chatbot ---
@st.cache_resource
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7) # Model 3.5-turbo
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

conversation_chain = get_conversation_chain(vectorstore)

# --- Inisialisasi Riwayat Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Tampilkan Riwayat Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input Pengguna ---
if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            response = conversation_chain({"question": prompt})
            st.markdown(response["chat_history"][-1].content) # Ambil respons terbaru dari chat_history
    st.session_state.messages.append({"role": "assistant", "content": response["chat_history"][-1].content})

st.markdown("---")
st.info("Catatan: Chatbot ini menggunakan Retrieval-Augmented Generation (RAG). Jawaban didasarkan pada dokumen yang Anda unggah dan kemampuan generatif dari OpenAI GPT-3.5-turbo.")
