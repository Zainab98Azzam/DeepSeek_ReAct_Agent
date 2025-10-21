# chat_ui.py
import streamlit as st
import requests

# Set the URL of your FastAPI backend
FASTAPI_URL = "http://127.0.0.1:8000/chat"

# Set up the Streamlit UI elements
st.title("Medical Agent Chat")

# Initialize chat history in Streamlit's session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages from the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input from the chat box
if user_input := st.chat_input("How can I help you?"):
    # Add the user's message to the chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the FastAPI backend to get a response from the agent
    with st.spinner("Thinking..."):
        try:
            response = requests.post(FASTAPI_URL, json={"query": user_input})

            if response.status_code == 200:
                # Extract the agent's final answer from the response
                agent_response = response.json().get("response", "No response found.")
                
                # Add the agent's response to the chat history and display it
                st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
                with st.chat_message("assistant"):
                    st.markdown(agent_response)
            else:
                # Display an error message if the backend request fails
                st.error(f"Error from backend: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: The backend server is not running or is unreachable.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")