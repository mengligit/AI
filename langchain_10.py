"""Example 2: Building chat prompts with ChatPromptTemplate."""

import os
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def main() -> None:
    """
    This example demonstrates several common usages of `ChatPromptTemplate`:
    1. Directly using a list of message types and template strings.
    2. Using `MessagePromptTemplate` subclasses to build more structured prompts.
    3. Combining prompts with a `ChatModel` into an LCEL chain to complete one call.
    """

    # --- 1. Create a ChatPromptTemplate using a list of tuples ---
    print("--- 1. Create using a list of tuples ---")

    chat_template_1 = ChatPromptTemplate.from_messages([
        ("system", "You are a professional assistant skilled at writing copy in a {style} style."),
        ("human", "Write a slogan for the product {product}."),
    ])

    print(f"Input variables required by chat template 1: {chat_template_1.input_variables}")

    formatted_messages_1 = chat_template_1.format_messages(
        style="humorous",
        product="automatic dishwasher",
    )

    print("\nFormatted message list 1:")
    for msg in formatted_messages_1:
        print(f" - Type: {type(msg).__name__}, Content: '{msg.content}'")
    print("-" * 30)

    # --- 2. Create using a list of MessagePromptTemplate objects ---
    print("\n--- 2. Create using MessagePromptTemplate objects ---")

    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a professional assistant skilled at writing copy in a {style} style."
    )
    human_prompt = HumanMessagePromptTemplate.from_template(
        "Write a slogan for the product {product}."
    )

    chat_template_2 = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    formatted_messages_2 = chat_template_2.format_messages(
        style="formal",
        product="high-end watch",
    )

    print("\nFormatted message list 2:")
    for msg in formatted_messages_2:
        print(f" - Type: {type(msg).__name__}, Content: '{msg.content}'")
    print("-" * 30)

    # --- 3. Use in an LCEL chain ---
    print("\n--- 3. Use in an LCEL chain ---")


    try:
        from langchain_deepseek import ChatDeepSeek
        model = ChatDeepSeek(model="deepseek-chat", api_key="sk-xxx")

        chain = chat_template_1 | model | StrOutputParser()

        chain_inputs = {"style": "science fiction", "product": "flying car"}
        print(f"Chain input: {chain_inputs}")

        response_text = chain.invoke(chain_inputs)

        print("\nChain output:")
        print(response_text)
    except Exception as exc:
        print(f"Failed to execute the LCEL chain: {exc}")


if __name__ == "__main__":
    # pip install langchain langchain-openai python-dotenv
    main()
