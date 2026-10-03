import asyncio
import sys
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from ddgs import DDGS  # Updated import statement
from typing import Annotated

# Load environment variables (.env)
load_dotenv()

def get_weather_news(city: Annotated[str, "The name of the city to get weather news for"]) -> str:
    """
    Search the web for the latest, up-to-date weather news and forecasts for a given city today.
    Use this tool exclusively for today's current weather events, not for general geography or historical trivia.
    """
    with DDGS() as ddgs:
        results = ddgs.text(f"{city} current weather news today", max_results=3)
        if not results:
            return f"No recent weather news found for {city}."
        
        return "\n".join([result['body'] for result in results])

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    agent = AssistantAgent(
        name="WeatherNewsAgent",
        model_client=model_client,
        tools=[get_weather_news],
        system_message="You are a weather news reporter. When asked for weather news, use your tool to search the web, then always format your final response as exactly 3 short lines.",
        reflect_on_tool_use=True
    )
    
    print("--- Weather News: Tokyo ---")
    result_tokyo = await agent.run(task="What is the weather news for Tokyo today?")
    print(result_tokyo.messages[-1].content)
    
    print("\n--- Weather News: Paris ---")
    result_paris = await agent.run(task="What is the weather news for Paris today?")
    print(result_paris.messages[-1].content)

if __name__ == "__main__":
    # Fix for the unclosed transport ResourceWarning on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())