import os
import time  # For a slightly better UX with streaming-like effect

import requests  # To make HTTP requests to our FastAPI backend
import streamlit as st

# --- Configuration ---
FASTAPI_URL = "http://localhost:8000/ask"  # URL of your FastAPI backend
POLICY_DOCS_PATH = "policies"  # For the admin uploader


# --- Helper Function for Admin: Policy Ingestion Trigger ---
# This is a simplified approach. In a more robust app, the FastAPI backend
# would handle the ingestion process via an API endpoint.
# For now, uploading a file here will require you to MANUALLY re-run ingest.py.
def trigger_ingestion():
    # This function is a placeholder.
    # Ideally, this would call an API endpoint on your FastAPI backend
    # that then runs the ingestion logic.
    # For now, it just reminds the user.
    st.sidebar.info(
        "Policy file uploaded. Please re-run `python ingest.py` in your terminal to update the knowledge base."
    )
    # Example of how you might try to run it (but this has limitations in Streamlit's cloud environment):
    # try:
    #     st.sidebar.write("Attempting to re-run ingestion (experimental)...")
    #     result = subprocess.run(["python", "ingest.py"], capture_output=True, text=True, check=True)
    #     st.sidebar.success("Ingestion script ran.")
    #     st.sidebar.code(result.stdout)
    #     if result.stderr:
    #         st.sidebar.error(result.stderr)
    # except Exception as e:
    #     st.sidebar.error(f"Could not auto-run ingest.py: {e}")
    #     st.sidebar.warning("Please run `python ingest.py` manually from your project's venv terminal.")


# --- Streamlit App Layout ---
st.set_page_config(page_title="HR AI Agent", layout="wide")

st.title("🏢 HR AI Agent Chatbot")
st.caption("Ask me anything about company policies!")

# --- Admin Section (Sidebar) for Policy Upload ---
with st.sidebar:
    st.header("⚙️ Admin Panel")
    st.subheader("Upload New Policies")
    uploaded_file = st.file_uploader(
        "Upload policy documents (.txt, .pdf, .md)",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,  # Allow uploading multiple files at once
    )

    if uploaded_file:
        all_successful = True
        for file in uploaded_file:
            try:
                # Save the file to the policies directory
                # Ensure the 'policies' directory exists
                os.makedirs(POLICY_DOCS_PATH, exist_ok=True)
                file_path = os.path.join(POLICY_DOCS_PATH, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.sidebar.success(
                    f"'{file.name}' uploaded successfully to '{POLICY_DOCS_PATH}/'."
                )
            except Exception as e:
                st.sidebar.error(f"Error uploading '{file.name}': {e}")
                all_successful = False
        if all_successful:
            trigger_ingestion()  # Remind user to re-run ingest.py

    st.markdown("---")
    st.subheader("ℹ️ System Info")
    st.markdown(f"**Knowledge Base:** `{POLICY_DOCS_PATH}/`")
    st.markdown(f"**Vector DB:** `chroma_db_hr/`")
    st.markdown(f"**FastAPI Backend:** `{FASTAPI_URL}`")


# --- Chat Interface ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you with company policies today?",
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):
            with st.expander("Sources Used"):
                for i, source in enumerate(message["sources"]):
                    st.caption(
                        f"Source {i+1}: {source['metadata'].get('source', 'N/A')}"
                    )
                    st.markdown(
                        f"```text\n{source['page_content'][:300]}...\n```"
                    )  # Show snippet

# React to user input
if prompt := st.chat_input("Ask your question here..."):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        source_documents_for_display = []

        try:
            # Prepare payload for FastAPI
            payload = {"query": prompt}
            response = requests.post(
                FASTAPI_URL, json=payload, timeout=60
            )  # Increased timeout
            response.raise_for_status()  # Raise an exception for HTTP errors (4XX or 5XX)

            api_response_data = response.json()
            full_response_content = api_response_data.get(
                "answer", "Sorry, I couldn't get an answer from the backend."
            )
            source_documents_for_display = api_response_data.get("source_documents", [])

            # Simulate stream of response with milliseconds delay for better UX
            # (Real streaming would require backend changes)
            for chunk in full_response_content.split():
                message_placeholder.markdown(
                    full_response_content[: len(chunk)] + "▌"
                )  # Show current part
                time.sleep(0.05)
            message_placeholder.markdown(full_response_content)  # Display full message

            if source_documents_for_display:
                with st.expander("Sources Used"):
                    for i, source in enumerate(source_documents_for_display):
                        st.caption(
                            f"Source {i+1}: {source['metadata'].get('source', 'N/A')}"
                        )
                        st.markdown(
                            f"```text\n{source['page_content'][:300]}...\n```"
                        )  # Show snippet

        except requests.exceptions.Timeout:
            full_response_content = (
                "Sorry, the request to the AI backend timed out. Please try again."
            )
            message_placeholder.error(full_response_content)
        except requests.exceptions.ConnectionError:
            full_response_content = f"Sorry, I couldn't connect to the AI backend at {FASTAPI_URL}. Is it running?"
            message_placeholder.error(full_response_content)
        except requests.exceptions.RequestException as e:
            full_response_content = (
                f"An error occurred while communicating with the backend: {e}"
            )
            message_placeholder.error(full_response_content)
        except Exception as e:
            full_response_content = f"An unexpected error occurred: {e}"
            message_placeholder.error(full_response_content)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response_content,
            "sources": source_documents_for_display,  # Store sources with the message
        }
    )
