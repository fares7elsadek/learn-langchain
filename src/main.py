
import os
import requests

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from helper.config import get_settings


settings = get_settings()
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY


# Information supplied by your application
@dataclass
class Context:
    user_id: str


# Shape of the final answer
@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool("locate_user")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    """Find the city associated with the current user's ID."""

    user_id = runtime.context.user_id

    match user_id:
        case "1":
            return "Cairo"
        case "2":
            return "Makkah"
        case _:
            return "Unknown"


@tool("get_weather")
def get_weather(city: str) -> dict:
    """Get the current weather conditions for a city."""

    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    current = data["current_condition"][0]

    return {
        "city": city,
        "condition": current["weatherDesc"][0]["value"],
        "temperature_celsius": float(current["temp_C"]),
        "temperature_fahrenheit": float(current["temp_F"]),
        "humidity": float(current["humidity"]),
    }


checkpointer = InMemorySaver()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
)

agent = create_agent(
    model=model,
    tools=[locate_user, get_weather],
    system_prompt=(
        "You are a helpful weather assistant. "
        "When the user asks about the weather in their location, "
        "first call locate_user to find their city. "
        "Then call get_weather with that city. "
        "Use the returned weather data for your answer. "
        "If the location is unknown, ask the user for their city."
    ),
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather now?",
            }
        ]
    },
    config=config,
    context=Context(user_id="2"),
)

print(response["structured_response"])
