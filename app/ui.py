import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 60


st.set_page_config(page_title="Агентный помощник", page_icon="🤖")
st.title("🤖 Личный Ассистент (Agentic RAG)")


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _upload_document(uploaded_file) -> None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def _ask_assistant(prompt: str) -> str:
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"text": prompt},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["answer"]


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Загрузка документов")
        uploaded_file = st.file_uploader("Загрузите PDF или TXT", type=["pdf", "txt"])

        if uploaded_file and st.button("Индексировать файл"):
            with st.spinner("Обработка..."):
                try:
                    _upload_document(uploaded_file)
                    st.success("Файл добавлен в базу знаний!")
                except requests.RequestException as exc:
                    st.error(f"Ошибка при загрузке: {exc}")


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _handle_user_prompt() -> None:
    if prompt := st.chat_input("Спросите что-нибудь..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    answer = _ask_assistant(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except requests.RequestException as exc:
                    st.error(f"Не удалось связаться с сервером: {exc}")


_init_session_state()
_render_sidebar()
_render_chat_history()
_handle_user_prompt()
