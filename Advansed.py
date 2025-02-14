import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import fitz
import logging
import requests

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Импорт кеширования через LLM
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Настраиваем кеш (результаты запросов LLM будут сохраняться в памяти)
set_llm_cache(InMemoryCache())

logging.basicConfig(level=logging.INFO)


# Telegram настройки

TELEGRAM_BOT_TOKEN = "7747155220:AAGYM6f6OePOmFUrWX5u9fVvuIydY7qKB-c"
TELEGRAM_CHAT_ID = st.text_input("Введите ваш Telegram Chat ID:")

def send_telegram_message_sync(chat_id, message):
    """Синхронная отправка сообщения в Telegram через HTTP-запрос."""
    max_length = 4096
    if len(message) > max_length:
        message = message[:max_length - 10] + "..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            logging.error(f"Ошибка отправки в Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Ошибка отправки в Telegram: {e}")


# Callback для Telegram с фильтрацией служебной информации

class TelegramCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        answer_text = ""
        if isinstance(response, dict):
            if "answer" in response:
                answer_text = response["answer"]
            elif "generations" in response:
                gens = response["generations"]
                if isinstance(gens, list) and len(gens) > 0:
                    first_chunk_list = gens[0]
                    if isinstance(first_chunk_list, list) and len(first_chunk_list) > 0:
                        answer_text = first_chunk_list[0].text
                    else:
                        answer_text = str(gens)
                else:
                    answer_text = str(response)
            else:
                answer_text = str(response)
        else:
            answer_text = str(response)
        
        if TELEGRAM_CHAT_ID:
            message = f"📢 LLM завершил обработку запроса!\n\nОтвет: {answer_text}"
            send_telegram_message_sync(TELEGRAM_CHAT_ID, message)


# Фильтрация нецензурных слов через LangChain

def filter_swear_words_langchain(text: str) -> str:
    template = (
        "Проверь следующий текст на наличие нецензурных слов.\n"
        "Список запрещённых слов: дурак, идиот, тупой, урод, козел, сволочь, idiot, stupid, shit, bastard\n"
        "Если текст содержит одно или более запрещённых слов, выведи только 'BAD_WORD_FOUND'.\n"
        "Если текст не содержит нецензурных слов, выведи только 'OK'.\n"
        "Текст: {text}"
    )
    prompt = PromptTemplate(input_variables=["text"], template=template)
    chain = prompt | OllamaLLM(model="llama3.2:latest")
    response = chain.invoke({"text": text}).strip()
    if "BAD_WORD_FOUND" in response:
        if TELEGRAM_CHAT_ID:
            send_telegram_message_sync(TELEGRAM_CHAT_ID, "⚠️ Ай-ай-ай, так нельзя писать плохие слова!")
        return None
    return text


# Извлечение текста из PDF

def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_document])
        return text.strip()
    except Exception as e:
        logging.error(f"Ошибка при обработке PDF: {e}")
        return None


# Создание векторного хранилища из текста

def create_vector_store(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)
    embeddings = OllamaEmbeddings(model="llama3.2:latest")
    vector_store = Chroma.from_texts(texts, embedding=embeddings, collection_name="pdf_collection")
    return vector_store


# Основная функция приложения

def main():
    st.title("📄 PDF-Анализатор")

    if not TELEGRAM_CHAT_ID:
        st.error("Пожалуйста, введите ваш Telegram Chat ID для корректной работы уведомлений.")
        st.stop()

    # Загрузка PDF-файла
    uploaded_file = st.file_uploader("Загрузите PDF-файл", type=["pdf"])
    pdf_text = extract_text_from_pdf(uploaded_file) if uploaded_file else None
    
    user_input = st.text_input("Введите ваш запрос:")

    # Если в session_state ещё нет буфера, создаём его (для сохранения контекста между запросами)
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Размещаем кнопки: отправка запроса, очистка кеша и очистка буфера
    col1, col2, col3 = st.columns(3)
    send_query = col1.button("Отправить запрос")
    clear_cache = col2.button("Очистить кеш LLM")
    clear_buffer = col3.button("Очистить буфер")

    if clear_cache:
        # Пересоздаем кеш, тем самым очищая его
        set_llm_cache(InMemoryCache())
        st.success("Кеш LLM очищен!")

    if clear_buffer:
        # Очищаем буфер диалога (краткосрочная память)
        st.session_state.conversation_memory.clear()
        st.success("Буфер очищен!")

    if send_query:
        if not user_input.strip():
            st.warning("Пожалуйста, введите запрос.")
            return
        if not pdf_text:
            st.warning("Файл не содержит текста или не был загружен.")
            return

        filtered_input = filter_swear_words_langchain(user_input)
        if filtered_input is None:
            return

        with st.spinner("Идет обработка PDF и создание векторного хранилища..."):
            vector_store = create_vector_store(pdf_text)
        
        llm = OllamaLLM(model="llama3.2:latest", callbacks=[TelegramCallbackHandler()])
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vector_store.as_retriever(),
            memory=st.session_state.conversation_memory
        )
        
        st.write("📢 Генерация ответа через LLM...")
        with st.spinner("Генерация ответа..."):
            response = qa_chain({"question": filtered_input})
        
        answer = response.get("answer", "Нет ответа")
        st.markdown("### ✅ Ответ:")
        st.write(answer)
        
        if TELEGRAM_CHAT_ID:
            send_telegram_message_sync(TELEGRAM_CHAT_ID, f"✅ Ответ на ваш запрос:\n\n{answer}")
            
if __name__ == "__main__":
    main()
