import nest_asyncio
import streamlit as st
import fitz
import telegram
import logging
import asyncio

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

logging.basicConfig(level=logging.INFO)

# Ввод Telegram Chat ID пользователем
TELEGRAM_CHAT_ID = st.text_input("Введите ваш Telegram Chat ID:", type="password")

def send_telegram_message(chat_id, message):
    try:
        bot = telegram.Bot(token="7747155220:AAGYM6f6OePOmFUrWX5u9fVvuIydY7qKB-c")
        asyncio.run(bot.send_message(chat_id=chat_id, text=message))
    except Exception as e:
        logging.error(f"Ошибка отправки в Telegram: {e}")

# Callback для уведомлений в Telegram
class TelegramCallbackHandler(BaseCallbackHandler):
    async def on_llm_end(self, response, **kwargs):
        if TELEGRAM_CHAT_ID:
            message = f"📢 LLM завершил обработку запроса!\n\nОтвет: {response}"
            send_telegram_message(TELEGRAM_CHAT_ID, message)

async def filter_swear_words(text):
    swear_words = ['fuck you', 'fuck','блядь','pidoras', 'nigga', 'shit','bitch', 'idiot', 'slut','bastard', 'asshole']
    found_bad_words = [word for word in swear_words if word in text.lower()]
    if found_bad_words and TELEGRAM_CHAT_ID:
        message = f"⚠️ В запросе найдены нецензурные слова: {', '.join(found_bad_words)}"
        send_telegram_message(TELEGRAM_CHAT_ID, message)
    return text

def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_document])
        return text.strip()
    except Exception as e:
        logging.error(f"Ошибка при обработке PDF: {e}")
        return None

def main():
    st.title("📄 PDF-Анализатор с Llama3 и Telegram Callback")
    uploaded_file = st.file_uploader("Загрузите PDF-файл", type=["pdf"])
    pdf_text = extract_text_from_pdf(uploaded_file) if uploaded_file else None
    user_input = st.text_input("Введите ваш запрос:")
    if st.button("Отправить запрос"):
        if not user_input.strip():
            st.warning("Пожалуйста, введите запрос.")
            return
        if not pdf_text:
            st.warning("Файл не содержит текста или не был загружен.")
            return
        filtered_input = asyncio.run(filter_swear_words(user_input))
        full_input = f"{pdf_text}\n\nВопрос: {filtered_input}"
        st.write("📢 Генерация ответа через LLM...")
        llm = OllamaLLM(model="llama3.2:latest", callbacks=[TelegramCallbackHandler()])
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
            if TELEGRAM_CHAT_ID:
                send_telegram_message(TELEGRAM_CHAT_ID, f"✅ Ответ на ваш запрос:\n\n{response}")
        except Exception as e:
            st.error(f"Ошибка генерации ответа: {e}")

if name == "main":
    main()
