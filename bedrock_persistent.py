import pickle
import os
import uuid
import logging
from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from threading import Lock
import chainlit as cl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable for the system prompt
system_prompt = ""

# Create a persistent memory dictionary with concurrency protection using a lock
persistent_memory = {}
memory_lock = Lock()

# Load any existing conversations from persistent memory
if os.path.exists("persistent_memory.pkl"):
    with open("persistent_memory.pkl", "rb") as f:
        persistent_memory = pickle.load(f)

def get_user_id():
    """
    Get or generate a unique identifier for the user session.
    """
    user_id = cl.user_session.get("user_id")
    if user_id is None:
        user_id = str(uuid.uuid4())
        cl.user_session.set("user_id", user_id)
    return user_id

def create_runnable():
    """
    Create and store a runnable in the session.
    """
    global system_prompt
    model = BedrockChat(
        credentials_profile_name="bedrock",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    return runnable

@cl.on_chat_start
async def on_chat_start():
    """
    Handle the start of a chat session.
    """
    await cl.Message(content="Hello there, I am Claude Sonnet from AWS Bedrock. How can I help you?").send()
    await cl.Message(content="Please enter the system prompt:").send()

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and commands.
    """
    global system_prompt
    user_id = get_user_id()

    # Ensure the system prompt is set
    if not system_prompt:
        system_prompt = message.content.strip()
        if not system_prompt:
            await cl.Message(content="System prompt cannot be empty. Please enter the system prompt:").send()
            return
        await cl.Message(content=f"System prompt set to: {system_prompt}").send()
        create_runnable()
        return

    # Validate user input
    user_input = message.content.strip()
    if not user_input:
        await cl.Message(content="Your input is empty. Please enter a valid message.").send()
        return

    # Handle regular messages and commands
    if user_input.lower().startswith("reset"):
        await on_reset_command(user_id)
    elif user_input.lower().startswith("history"):
        await on_history_command(user_id)
    else:
        await handle_user_message(user_input, user_id)

async def handle_user_message(user_input: str, user_id):
    """
    Handle user messages and generate responses.
    """
    try:
        with memory_lock:
            if user_id not in persistent_memory:
                persistent_memory[user_id] = []

            # Retrieve the conversation history
            conversation_history = persistent_memory[user_id]

            # Add the new message to the conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Prepare the context including the conversation history
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
            context += f"\nuser: {user_input}"

            runnable = cl.user_session.get("runnable")
            if runnable is None:
                runnable = create_runnable()

            msg = cl.Message(content="")

            async for chunk in runnable.astream(
                {"question": context},
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
            ):
                await msg.stream_token(chunk)

            await msg.send()

            # Add the assistant's response to the conversation history
            conversation_history.append({"role": "assistant", "content": msg.content})

        # Save the updated memory to persistent storage
        with open("persistent_memory.pkl", "wb") as f:
            pickle.dump(persistent_memory, f)

    except Exception as e:
        logging.error(f"Error handling user message: {e}")
        await cl.Message(content="An error occurred while processing your message. Please try again.").send()

async def on_history_command(user_id):
    """
    Retrieve and display the conversation history.
    """
    try:
        with memory_lock:
            conversation_history = persistent_memory.get(user_id, [])
            history_content = "\n".join([f"{msg['role']}: {msg['content']}"] for msg in conversation_history)
            await cl.Message(content=f"Conversation history:\n{history_content}").send()
    except Exception as e:
        logging.error(f"Error retrieving conversation history: {e}")
        await cl.Message(content="An error occurred while retrieving the conversation history.").send()

async def on_reset_command(user_id):
    """
    Reset the state for the user session.
    """
    try:
        with memory_lock:
            if user_id in persistent_memory:
                del persistent_memory[user_id]
                await cl.Message(content="State reset successfully!").send()
    except Exception as e:
        logging.error(f"Error resetting state: {e}")
        await cl.Message(content="An error occurred while resetting the state.").send()
