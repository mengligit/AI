import os
import json
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- 1. Define the same tool as in example_1 ---
class GetWeatherArgs(BaseModel):
    location: str = Field(description="Name of the city to query weather for, e.g. Beijing")


@tool(args_schema=GetWeatherArgs)
def get_weather(location: str) -> str:
    """Used to get current weather information for a specified city."""
    if "Beijing" in location:
        return json.dumps({"location": "Beijing", "temperature": "25°C", "condition": "Sunny"})
    elif "Shanghai" in location:
        return json.dumps({"location": "Shanghai", "temperature": "28°C", "condition": "Cloudy"})
    else:
        return json.dumps({"location": location, "temperature": "Unknown", "condition": "Unknown"})


def main():
    """
    This example demonstrates a complete end-to-end function calling flow:
    1. The model decides to call a tool.
    2. We parse that call and execute the corresponding function.
    3. We wrap the function's return result in a `ToolMessage`.
    4. We send the `ToolMessage` together with the original conversation history to the model again.
    5. The model generates the final natural language answer based on the tool's return result.
    """

    from langchain_deepseek import ChatDeepSeek
    model = ChatDeepSeek(model="deepseek-chat", api_key="sk-cff8a797d9874412a529768e54c13f57")

    model_with_tools = model.bind_tools([get_weather])

    messages = [HumanMessage(content="What's the weather like in Shanghai today?")]

    print("--- Step 1 & 2: Model decision ---")
    first_response = model_with_tools.invoke(messages)
    messages.append(first_response)

    print(f"The AIMessage returned by the model contains tool_calls: {first_response.tool_calls}")
    print("-" * 30)

    print("\n--- Step 3 & 4: Execute tools on the application side ---")
    if not first_response.tool_calls:
        print("The model did not call any tools. Ending the flow.")
        return

    for tool_call in first_response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"Detected tool call: {tool_name}({tool_args})")

        if tool_name == get_weather.name:
            tool_output = get_weather.invoke(tool_args)
            print(f"Tool execution result: {tool_output}")
            messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call_id))
        else:
            print(f"Unknown tool: {tool_name}")
            messages.append(ToolMessage(content=f"Error: unknown tool '{tool_name}'", tool_call_id=tool_call_id))

    print("\nCurrent message history:")
    pprint(messages)
    print("-" * 30)

    print("\n--- Step 5: Model generates final answer ---")
    final_response = model_with_tools.invoke(messages)

    print("\n--- Final natural language answer ---")
    print(f"Type: {type(final_response)}")
    print(f"Content: {final_response.content}")
    print("-" * 30)

    print("\nSummary: This 'model -> tool -> model' loop is the core of building Agents that can interact with the external world.")


if __name__ == "__main__":
    main()