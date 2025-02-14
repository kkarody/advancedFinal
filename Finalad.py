import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import fitz
import telegram
import logging
import asyncio

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = "7747155220:AAGYM6f6OePOmFUrWX5u9fVvuIydY7qKB-c"
TELEGRAM_CHAT_ID = "392539749"

# Callback для уведомлений в Telegram
class TelegramCallbackHandler(BaseCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        info_message = "📢 Telegram Callback вызван!"
        st.write(info_message)
        logging.info(info_message)
        try:
            local_bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            await local_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=info_message)
        except Exception as e:
            error_message = f"❌ Ошибка отправки Telegram: {e}"
            logging.error(error_message)
            st.error(error_message)

# Функция для фильтрации нецензурных слов
async def filter_swear_words(text):
    logging.info(f"📢 Фильтрация текста: {text}")
    swear_words = ['badword1', 'badword2', 'badword3']
    found_bad_words = [word for word in swear_words if word in text.lower()]
    if found_bad_words:
        message = f"⚠️ В запросе найдены нецензурные слова: {', '.join(found_bad_words)}"
        logging.info(f"📢 Отправка Telegram: {message}")
        try:
            local_bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            await local_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            logging.error(f"❌ Ошибка отправки Telegram: {e}")
    return text

vector_store = Chroma(
    persist_directory="db_store",
    embedding_function=OllamaEmbeddings(model="llama3.2:latest")
)

def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_document])
        return text.strip()
    except Exception as e:
        error_message = f"Ошибка при обработке PDF: {e}"
        st.error(error_message)
        logging.error(error_message)
        return None

def main():
    st.title("📄 PDF-Анализатор с Llama3 и Telegram Callback")
    uploaded_file = st.file_uploader("Загрузите PDF-файл", type=["pdf"])
    pdf_text = None
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.text_area("📄 Извлечённый текст:", pdf_text[:2000], height=200)
    user_input = st.text_input("Введите ваш запрос:")
    if st.button("Отправить запрос"):
        if not user_input.strip():
            st.warning("Пожалуйста, введите запрос.")
            return
        if not pdf_text:
            st.warning("Файл не содержит текста или не был загружен.")
            return
        try:
            filtered_input = asyncio.run(filter_swear_words(user_input))
        except Exception as e:
            error_message = f"Ошибка при фильтрации текста: {e}"
            st.error(error_message)
            logging.error(error_message)
            filtered_input = user_input
        full_input = (
            f"Проанализируй следующий текст и ответь на вопрос:\n\n"
            f"{pdf_text}\n\n"
            f"Вопрос: {filtered_input}"
        )
        st.write("📢 Генерация ответа через LLM...")
        logging.info("📢 Генерация ответа через LLM...")
        llm = OllamaLLM(
            model="llama3.2:latest",
            callbacks=[TelegramCallbackHandler()]
        )
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="Дан текст: {input}\nПроанализируй текст и ответь на вопрос."
        )
        chain = prompt_template | llm
        try:
            with st.spinner("Генерация ответа..."):
                response = chain.invoke({"input": full_input})
            st.markdown("### ✅ Ответ:")
            st.write(response)
            logging.info(f"📢 Ответ LLM: {response}")
        except Exception as e:
            error_message = f"Ошибка генерации ответа: {e}"
            st.error(error_message)
            logging.error(error_message)

if __name__ == "__main__":
    main()
