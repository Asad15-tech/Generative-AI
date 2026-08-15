# ============================================================
# LANGCHAIN AI ASSISTANT
# Professional Streamlit Chatbot UI
# ============================================================
"""
A production-grade Streamlit front-end for a LangChain-powered
learning assistant covering AI, Machine Learning, Python, and
Generative AI concepts.

Structure:
    - Configuration & static content are separated from rendering.
    - Every UI region is its own function (sidebar, hero, welcome,
      chat, footer) so the file stays readable and testable.
    - Errors from the model layer are caught, logged, and shown
      to the user without crashing the app.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Literal

import streamlit as st

from chatbot import get_response, clear_memory


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("langchain_ai_assistant")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AppConfig:
    """Static, immutable app-level configuration."""

    page_title: str = "LangChain AI Assistant"
    page_icon: str = "🤖"
    layout: Literal["centered", "wide"] = "wide"
    sidebar_state: Literal["auto", "expanded", "collapsed"] = "expanded"
    chat_input_placeholder: str = "✨ Ask me anything about AI, ML, Python or LangChain..."
    thinking_label: str = "🤔 Thinking..."


CONFIG = AppConfig()

FEATURES: list[dict[str, str]] = [
    {
        "icon": "🔗",
        "title": "LangChain",
        "text": "Learn prompts, models, chains, LCEL and other "
                "LangChain concepts through simple examples.",
    },
    {
        "icon": "🧠",
        "title": "AI & Machine Learning",
        "text": "Understand AI and Machine Learning concepts with "
                "practical explanations and examples.",
    },
    {
        "icon": "🐍",
        "title": "Python",
        "text": "Learn Python programming concepts with practical "
                "examples and easy-to-understand explanations.",
    },
]

TECHNOLOGIES: list[tuple[str, str]] = [
    ("🐍", "Python"),
    ("🔗", "LangChain"),
    ("🤗", "Hugging Face"),
    ("🎨", "Streamlit"),
]

CONCEPTS: list[tuple[str, str]] = [
    ("✅", "PromptTemplate"),
    ("✅", "ChatPromptTemplate"),
    ("✅", "MessagePlaceholder"),
    ("✅", "Few-Shot Prompting"),
    ("✅", "LCEL Pipeline"),
    ("✅", "Prompt Chaining"),
    ("🔜", "Conversation Memory"),
    ("🔜", "RAG"),
]

Role = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    """A single turn in the conversation."""

    role: Role
    content: str


# ============================================================
# CSS
# ============================================================

_CSS = textwrap.dedent("""
    <style>

    /* GLOBAL */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(14, 165, 233, 0.12), transparent 30%),
            linear-gradient(135deg, #f8faff 0%, #eef2ff 50%, #f0f9ff 100%);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e1b4b 45%, #312e81 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-logo {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .sidebar-logo-icon {
        font-size: 48px;
        margin-bottom: 5px;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #c7d2fe !important;
        margin-top: 4px;
    }

    /* SIDEBAR CARDS */
    .sidebar-card {
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 17px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }

    .sidebar-card-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .sidebar-card-text {
        font-size: 13px;
        line-height: 1.7;
        color: #dbeafe !important;
    }

    .concept-item {
        padding: 5px 0;
        font-size: 13px;
        color: #e0e7ff !important;
    }

    /* HERO */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 42px 45px;
        margin-bottom: 30px;
        border-radius: 26px;
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 45%, #0891b2 100%);
        box-shadow: 0 20px 45px rgba(79,70,229,0.25);
        color: white;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        background: rgba(255,255,255,0.08);
        right: -70px;
        top: -90px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 8px;
        color: white;
    }

    .hero-subtitle {
        font-size: 16px;
        line-height: 1.7;
        color: #e0e7ff;
        max-width: 700px;
    }

    .online-badge {
        display: inline-block;
        margin-top: 20px;
        padding: 8px 15px;
        border-radius: 30px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        font-size: 13px;
        font-weight: 600;
        color: white;
    }

    /* WELCOME */
    .welcome {
        text-align: center;
        padding: 30px 20px 25px 20px;
    }

    .welcome-icon {
        font-size: 52px;
        margin-bottom: 8px;
    }

    .welcome-title {
        font-size: 30px;
        font-weight: 800;
        color: #1e1b4b;
        margin-bottom: 8px;
    }

    .welcome-text {
        font-size: 15px;
        color: #64748b;
    }

    /* FEATURE CARDS */
    .feature-card {
        height: 100%;
        min-height: 170px;
        padding: 25px;
        border-radius: 20px;
        background: rgba(255,255,255,0.9);
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
        transition: all 0.25s ease;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 35px rgba(79,70,229,0.12);
        border-color: #c7d2fe;
    }

    .feature-icon {
        font-size: 30px;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 17px;
        font-weight: 750;
        color: #1e1b4b;
        margin-bottom: 8px;
    }

    .feature-text {
        font-size: 13px;
        line-height: 1.7;
        color: #64748b;
    }

    /* SECTION TITLE */
    .section-title {
        font-size: 20px;
        font-weight: 750;
        color: #1e1b4b;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* CHAT AREA */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 5px;
        margin-bottom: 10px;
    }

    [data-testid="stChatMessage"] p {
        font-size: 15px;
        line-height: 1.7;
    }

    /* CHAT INPUT */
    [data-testid="stChatInput"] {
        border-radius: 18px;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 15px !important;
        border: 2px solid #c7d2fe !important;
        background: white !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* BUTTONS */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.08);
        color: white;
        font-weight: 650;
        padding: 10px;
    }

    .stButton > button:hover {
        background: rgba(255,255,255,0.16);
        border-color: rgba(255,255,255,0.3);
    }

    /* DIVIDER */
    .custom-divider {
        height: 1px;
        background: rgba(255,255,255,0.15);
        margin: 20px 0;
    }

    /* FOOTER */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-size: 12px;
        color: #94a3b8;
    }

    </style>
""")


def load_css() -> None:
    """Inject the app's custom stylesheet."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    """Ensure required keys exist in Streamlit's session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # type: list[ChatMessage]


def get_messages() -> list[ChatMessage]:
    return st.session_state.messages


def add_message(role: Role, content: str) -> None:
    st.session_state.messages.append(ChatMessage(role=role, content=content))


def clear_messages() -> None:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            textwrap.dedent("""
            <div class="sidebar-logo">
                <div class="sidebar-logo-icon">🤖</div>
                <div class="sidebar-title">LangChain AI</div>
                <div class="sidebar-subtitle">AI Learning Assistant</div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        _render_about_card()
        _render_technologies_card()
        _render_concepts_card()

        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_messages()
            clear_memory()
            st.rerun()


def _render_about_card() -> None:
    st.markdown("### 🧠 About")
    st.markdown(
        textwrap.dedent("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">AI Learning Assistant</div>
            <div class="sidebar-card-text">
                Learn Artificial Intelligence, Machine Learning, Python,
                Generative AI and LangChain through an interactive AI assistant.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def _render_technologies_card() -> None:
    st.markdown("### ⚡ Technologies")
    items = "<br>\n".join(f"{icon} <b>{name}</b>" for icon, name in TECHNOLOGIES)
    st.markdown(
        f'<div class="sidebar-card"><div class="sidebar-card-text">{items}</div></div>',
        unsafe_allow_html=True,
    )


def _render_concepts_card() -> None:
    st.markdown("### 🔗 LangChain Concepts")
    items = "\n".join(
        f'<div class="concept-item">{status} {name}</div>' for status, name in CONCEPTS
    )
    st.markdown(f'<div class="sidebar-card">{items}</div>', unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

def render_hero() -> None:
    st.markdown(
        textwrap.dedent(f"""
        <div class="hero">
            <div class="hero-title">🤖 {CONFIG.page_title}</div>
            <div class="hero-subtitle">
                Your intelligent learning companion for Artificial Intelligence,
                Machine Learning, Python, Generative AI and LangChain.
            </div>
            <div class="online-badge">● Hugging Face Model Online</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ============================================================
# WELCOME SCREEN
# ============================================================

def render_welcome() -> None:
    st.markdown(
        textwrap.dedent("""
        <div class="welcome">
            <div class="welcome-icon">✨</div>
            <div class="welcome-title">How can I help you learn today?</div>
            <div class="welcome-text">
                Ask questions about AI, ML, Python, Generative AI or LangChain.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    columns = st.columns(len(FEATURES))
    for column, feature in zip(columns, FEATURES):
        with column:
            _render_feature_card(feature)


def _render_feature_card(feature: dict[str, str]) -> None:
    st.markdown(
        textwrap.dedent(f"""
        <div class="feature-card">
            <div class="feature-icon">{feature['icon']}</div>
            <div class="feature-title">{feature['title']}</div>
            <div class="feature-text">{feature['text']}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT
# ============================================================

def render_chat_history() -> None:
    for message in get_messages():
        with st.chat_message(message.role):
            st.markdown(message.content)


def generate_response(question: str) -> str:
    """Call the model layer and gracefully degrade on failure."""
    try:
        return get_response(question)
    except Exception:
        logger.exception("Failed to generate a response for: %r", question)
        return (
            "⚠️ **Something went wrong while generating a response.**\n\n"
            "Please try again in a moment."
        )


def handle_chat_input() -> None:
    user_question = st.chat_input(CONFIG.chat_input_placeholder)
    if not user_question:
        return

    add_message("user", user_question)
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner(CONFIG.thinking_label):
            response = generate_response(user_question)
        st.markdown(response)

    add_message("assistant", response)


# ============================================================
# FOOTER
# ============================================================

def render_footer() -> None:
    st.markdown(
        '<div class="footer">Built with Python • LangChain • Hugging Face • Streamlit</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# APP ENTRY POINT
# ============================================================

def main() -> None:
    st.set_page_config(
        page_title=CONFIG.page_title,
        page_icon=CONFIG.page_icon,
        layout=CONFIG.layout,
        initial_sidebar_state=CONFIG.sidebar_state,
    )

    load_css()
    init_session_state()

    render_sidebar()
    render_hero()

    if not get_messages():
        render_welcome()

    render_chat_history()
    handle_chat_input()
    render_footer()


if __name__ == "__main__":
    main()