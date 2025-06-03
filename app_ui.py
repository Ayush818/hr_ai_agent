# app_ui.py (Modifications)
import json  # For pretty printing dicts
import os
import time

import requests
import streamlit as st

# --- Configuration ---
FASTAPI_URL_CHAT = "http://localhost:8000/chat"  # Updated endpoint
POLICY_DOCS_PATH = "policies"


# ... (trigger_ingestion and Admin Sidebar can remain largely the same) ...
# --- Helper Function for Admin: Policy Ingestion Trigger ---
def trigger_ingestion():
    st.sidebar.info(
        "Policy file uploaded. Please re-run `python ingest.py` and restart the FastAPI backend to update the knowledge base."
    )


# --- Streamlit App Layout ---
st.set_page_config(page_title="HR AI Agent", layout="wide")

st.title("🏢 HR AI Agent Chatbot")
st.caption(
    "Ask policy questions or apply for leave (e.g., 'I want to apply for vacation next week from Monday to Friday')."
)

# --- Admin Section (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Admin Panel")
    st.subheader("Upload New Policies")
    uploaded_files = st.file_uploader(  # Renamed from uploaded_file
        "Upload policy documents (.txt, .pdf, .md)",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:  # Iterate if multiple files
        all_successful = True
        for file in uploaded_files:
            try:
                os.makedirs(POLICY_DOCS_PATH, exist_ok=True)
                file_path = os.path.join(POLICY_DOCS_PATH, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.sidebar.success(f"'{file.name}' uploaded to '{POLICY_DOCS_PATH}/'.")
            except Exception as e:
                st.sidebar.error(f"Error uploading '{file.name}': {e}")
                all_successful = False
        if (
            all_successful and uploaded_files
        ):  # Check if any files were actually uploaded
            trigger_ingestion()

    st.markdown("---")
    st.subheader("ℹ️ System Info")
    st.markdown(f"**Knowledge Base:** `{POLICY_DOCS_PATH}/`")
    st.markdown(f"**Vector DB:** `chroma_db_hr/`")
    st.markdown(f"**FastAPI Chat Endpoint:** `{FASTAPI_URL_CHAT}`")


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help with policies or leave requests?",
        }
    ]

if "current_leave_form" not in st.session_state:  # To store partially filled form data
    st.session_state.current_leave_form = {}


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "sources" in message and message["sources"]:
                with st.expander("Sources Used (Policy Q&A)"):
                    for i, source in enumerate(message["sources"]):
                        st.caption(
                            f"Source {i+1}: {source['metadata'].get('source', 'N/A')}"
                        )
                        st.markdown(f"```text\n{source['page_content'][:300]}...\n```")
            if (
                "data" in message
                and message["data"]
                and message.get("response_type") == "clarification_needed"
            ):
                with st.expander("Information Extracted So Far"):
                    st.json(message["data"])


if prompt := st.chat_input("What can I help you with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        response_data_for_display = None
        source_documents_for_display = None
        response_type = "error"  # Default

        try:
            payload = {"query": prompt, "user_id": 1}  # Hardcoding user_id=1 for now
            # Combine with any partially filled form data for context if needed (more advanced)
            # if st.session_state.current_leave_form:
            #    payload["contextual_form_data"] = st.session_state.current_leave_form

            api_response = requests.post(
                FASTAPI_URL_CHAT, json=payload, timeout=90
            )  # Increased timeout
            api_response.raise_for_status()

            response_json = api_response.json()
            response_type = response_json.get("response_type", "error")
            full_response_content = response_json.get(
                "message", "Sorry, an issue occurred."
            )
            response_data_for_display = response_json.get("data")
            source_documents_for_display = response_json.get("source_documents")

            # Simulate stream for UX
            # for chunk_word in full_response_content.split():
            #     message_placeholder.markdown(full_response_content[:len(chunk_word)] + "▌")
            #     time.sleep(0.05)
            message_placeholder.markdown(full_response_content)  # Display full message

            if response_type == "clarification_needed" and response_data_for_display:
                st.session_state.current_leave_form = (
                    response_data_for_display  # Store for next turn
                )
                with st.expander("Information Extracted So Far"):
                    st.json(response_data_for_display)
            elif response_type == "leave_submitted":
                st.session_state.current_leave_form = (
                    {}
                )  # Clear form on successful submission
                if (
                    response_data_for_display
                    and "request_id" in response_data_for_display
                ):
                    st.success(
                        f"Leave Request ID: {response_data_for_display['request_id']}"
                    )

            if source_documents_for_display and response_type == "policy_answer":
                with st.expander("Sources Used (Policy Q&A)"):
                    for i, source in enumerate(source_documents_for_display):
                        st.caption(
                            f"Source {i+1}: {source['metadata'].get('source', 'N/A')}"
                        )
                        st.markdown(f"```text\n{source['page_content'][:300]}...\n```")

        except requests.exceptions.Timeout:
            full_response_content = (
                "Sorry, the request to the AI backend timed out. Please try again."
            )
            message_placeholder.error(full_response_content)
        except requests.exceptions.ConnectionError:
            full_response_content = f"Sorry, I couldn't connect to the AI backend at {FASTAPI_URL_CHAT}. Is it running?"
            message_placeholder.error(full_response_content)
        except requests.exceptions.RequestException as e:
            full_response_content = f"Error communicating with backend: {e}. Response: {api_response.text if 'api_response' in locals() else 'N/A'}"
            message_placeholder.error(full_response_content)
        except Exception as e:
            full_response_content = f"An unexpected error occurred in UI: {e}"
            message_placeholder.error(full_response_content)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response_content,
            "sources": (
                source_documents_for_display
                if response_type == "policy_answer"
                else None
            ),
            "data": (
                response_data_for_display
                if response_type == "clarification_needed"
                else None
            ),
            "response_type": response_type,
        }
    )
