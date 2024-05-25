from langchain_community.chat_models import BedrockChat  # Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hello there, I am Claude Sonnet from AWS Bedrock. How can I help you?").send()

    # Updated to use BedrockChat
    model = BedrockChat(
        credentials_profile_name="bedrock", model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a very knowledgeable assistant. Please provide answers responsibly."),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
