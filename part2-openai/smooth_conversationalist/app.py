import os
from agents import Agent, Runner, trace
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")


smooth_conversation_agent = Agent(
    name="SmoothConversationAgent",
    instructions="You are a smooth conversationalist. You speak like James Bond.",
    model="gpt-4.1"
)


async def chat(message, history):

    clean_history = [{"role": m["role"], "content": m["content"]} for m in history]

    message = clean_history + [{'role':'user', 'content': message}]

    result = Runner.run_streamed(smooth_conversation_agent, message)
    content = ""
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            content += event.data.delta     
            yield content


import gradio as gr

with gr.ChatInterface(
    fn=chat,
    type="messages",
    description="Chat with a smooth conversational agent that speaks like James Bond.",
    theme="compact"
) as chat_interface:
    chat_interface.launch()