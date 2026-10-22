import os
import requests

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from helper.config import get_settings

settings = get_settings()
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY


@tool("get_weather")
def get_weather(city: str) -> dict:
    """Return weather information for a given city."""
    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in Cairo?"}
    ]
})

print(response["messages"][-1].content)