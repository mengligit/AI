from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# Initialize LLM
from langchain_deepseek import ChatDeepSeek
llm = ChatDeepSeek(model="deepseek-chat", api_key="sk-xxx")


# Routing function
def classify_intent(state: MessagesState) -> str:
    """Route to different Agents based on user intent."""
    last_message = state["messages"][-1]
    content = last_message.content.lower()

    if "weather" in content or "temperature" in content:
        return "weather_agent"
    elif "code" in content or "programming" in content:
        return "code_agent"
    elif "goodbye" in content or "exit" in content:
        return "farewell"
    else:
        return "general_agent"


# Define each Agent node
def router_node(state: MessagesState) -> dict:
    """Router node: does no processing, only used to trigger routing decision."""
    return {}


def weather_node(state: MessagesState) -> dict:
    """Weather Agent"""
    response = llm.invoke([
        SystemMessage(content="You are a weather assistant. Answer weather-related questions in a friendly way. If there is no real-time data, you can give general advice."),
        *state["messages"]
    ])
    return {"messages": [response]}


def code_node(state: MessagesState) -> dict:
    """Code Agent"""
    response = llm.invoke([
        SystemMessage(content="You are a programming assistant, skilled at answering code questions and providing clear code examples."),
        *state["messages"]
    ])
    return {"messages": [response]}


def general_node(state: MessagesState) -> dict:
    """General Agent"""
    response = llm.invoke([
        SystemMessage(content="You are a friendly AI assistant who can answer various questions."),
        *state["messages"]
    ])
    return {"messages": [response]}


def farewell_node(state: MessagesState) -> dict:
    """Farewell node"""
    return {"messages": [{"role": "assistant", "content": "Goodbye! Looking forward to talking with you next time."}]}


# Build graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("router", router_node)
builder.add_node("weather_agent", weather_node)
builder.add_node("code_agent", code_node)
builder.add_node("general_agent", general_node)
builder.add_node("farewell", farewell_node)

# Add edges
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    classify_intent,
    {
        "weather_agent": "weather_agent",
        "code_agent": "code_agent",
        "general_agent": "general_agent",
        "farewell": "farewell",
    }
)

# All agent nodes end after processing
for node in ["weather_agent", "code_agent", "general_agent", "farewell"]:
    builder.add_edge(node, END)

# Compile graph
graph = builder.compile()

# Test different intents
test_inputs = [
    "What's the weather in Beijing today?",
    "Help me write a Python quicksort",
    "Hello, introduce yourself",
    "Goodbye!"
]

for user_input in test_inputs:
    print(f"\nUser: {user_input}")
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    print(f"Assistant: {result['messages'][-1].content[:100]}...")
    print("-" * 50)
