import os
from tavily import TavilyClient
from dotenv import load_dotenv
import json
from openai import OpenAI
from utils import function_to_tool
import gradio as gr

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment variables.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

tavily_client = TavilyClient()
openai_client = OpenAI()


# All functions

def flight_search(query: str) -> str:
    """
    Searches for flights based on the provided query

    Args:
        query (str): The search query for flights

    Returns:
        str: A formatted string containing the search results
    
    """

    response = tavily_client.search(
    query=query,
    include_domains=["emirates.com","etihad.ae","skyscanner.com"]
    )

    results = response.get("results", [])

    # extract items
    contents = [item.get("content", "") for item in results]

    # Format as a numbered list
    formatted_contents = "\n".join(f"{i + 1}. {content}" for i, content in enumerate(contents) if content)

    return formatted_contents

def hotel_search(query: str) -> str:
    """
    Searches for hotels based on the provided query

    Args:
        query (str): The search query for hotels

    Returns:
        str: A formatted string containing the search results
    
    """

    response = tavily_client.search(
    query=query,
    include_domains=["booking.com","airbnb.com"]
    )

    results = response.get("results", [])

    # extract items
    contents = [item.get("content", "") for item in results]

    # Format as a numbered list
    formatted_contents = "\n".join(f"{i + 1}. {content}" for i, content in enumerate(contents) if content)

    return formatted_contents


def call_function(name, args):
    if name == "flight_search":
        return flight_search(**args)
    elif name == "hotel_search":
        return hotel_search(**args)
    else:
        raise ValueError(f"Unknown function: {name}")


def get_response(input_list):
    response = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=input_list,
        tools=[flight_tool_schema, hotel_tool_schema],
        tool_choice="auto",
        parallel_tool_calls=False
    )
    return response



# Class to define prompt template
class PromptTemplate:
    def __init__(self, template: str, input_variables: list[str]):
        self.template = template
        self.input_variables = input_variables

    def generate(self, **kwargs) -> str:
        return self.template.format(**{k: kwargs[k] for k in self.input_variables})


prompt = PromptTemplate(
    template="I want to travel to {destination} from {origin} on {departure_date} and return on {return_date}. I prefer {preferences}",
    input_variables=["destination", "origin", "departure_date", "return_date", "preferences"]
)




# Schemas
flight_tool_schema = function_to_tool(flight_search)
hotel_tool_schema = function_to_tool(hotel_search)


# App logic

system_message = """
You are an AI travel planner. The user will provide all their vacation details in a single prompt. Your job is to:

1. Use the provided details to search for the best flights and hotels using your tools.
2. Summarize the results in a neatly formatted, easy-to-read itinerary.
3. Do not ask the user any questions or request clarification—just use the information given.
4. Make the itinerary clear, helpful, and a little bit fun!

Example flow:
- Receive the user's vacation details (destination, dates, preferences, etc.).
- Use your flight tool to find the best flight options.
- Use your hotel search tools to find the best options.
- Present a summary itinerary like:

---
**Your Adventure Awaits!**
- Destination: Tokyo, Japan
- Dates: 2024-10-01 to 2024-10-10

✈️ **Flight Options:**
1. [Flight details here]

Day 1:
🏨 **Hotel Options:**
1. [Hotel details here]
2. List of things to do

Day 2:
🏨 **Hotel Options:**
1. [Hotel details here]
2. List of things to do

Day 3:
🏨 **Hotel Options:**
1. [Hotel details here]
2. List of things to do

Have a fantastic trip!
---

Only return the final itinerary—no follow-up questions or conversation needed.
Always call the flight tool first, then the hotel tool. Do not give me a response until you have called both tools.
"""

def trip_planner(destination, origin, departure_date, return_date, preferences):

    user_prompt = prompt.generate(
    destination=destination,
    origin=origin,
    departure_date=departure_date,
    return_date=return_date,
    preferences=preferences
    )

    input_list = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    response = get_response(input_list)

    if response.output[0].type == "function_call":
        while response.output[0].type == "function_call":
            input_list += response.output
            
            name = response.output[0].name
            args = json.loads(response.output[0].arguments)

            result = call_function(name, args)

            input_list.append({
                'type': 'function_call_output',
                'call_id': response.output[0].call_id,
                'output': str(result)
            })

            response = get_response(input_list)

    return response.output_text


# Gradio Interface

with gr.Blocks() as demo:
    gr.Markdown("# Atlas: Your Personal Travel Itinerary Planner")
    gr.Markdown("Plan your perfect trip with ease! Enter your travel details below, and let our AI-powered planner find the best flights and hotels for you.")
    
    with gr.Row():
        destination = gr.Textbox(label="Destination", placeholder="Enter your travel destination (e.g., Tokyo)")
        origin = gr.Textbox(label="Origin", placeholder="Enter your departure city (e.g., New York)")

    with gr.Row():
        departure_date = gr.Textbox(label="Departure Date", placeholder="Enter your departure date (e.g., 2024-10-01)")
        return_date = gr.Textbox(label="Return Date", placeholder="Enter your return date (e.g., 2024-10-10)")

    preferences = gr.Textbox(label="Preferences", placeholder="Enter any preferences (e.g., budget airlines, 4-star hotels)")
    submit_btn = gr.Button("Plan My Trip!")
    output = gr.Markdown(label="Your Itinerary")

    def on_submit(destination, origin, departure_date, return_date, preferences):
        return trip_planner(destination, origin, departure_date, return_date, preferences)
    
    submit_btn.click(on_submit, inputs=[destination, origin, departure_date, return_date, preferences], outputs=output)

demo.launch()