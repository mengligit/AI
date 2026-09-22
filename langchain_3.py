from langchain.agents import create_agent
from langchain.tools import tool

# Initialize a large model

from langchain_deepseek import ChatDeepSeek
model = ChatDeepSeek(model="deepseek-chat", api_key="sk-cff8a797d9874412a529768e54c13f57")

#@tool
def get_weather_err(city: str): #
    return f"{city} is cloudless"

@tool
def get_weather(city: str):
    """
       Get the current weather for a given city.
    """
    return f"{city} is cloudless"

# Create the agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# The agent starts calling the model
response = agent.invoke(
   dict(messages=[{"role":"user", "content": "What's the weather like in Beijing?"}])
)

print(response)
answer = response['messages'][-1].content
print(answer)
# Based on the query result, the current weather in Beijing is **cloudless**. This is a clear and nice day, suitable for outdoor activities.