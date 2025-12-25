# from langchain_ollama import ChatOllama
# from langchain.agents import create_agent
# from langchain.tools import tool
#
#
# @tool
# def summ_numbers(a: int, b: int) -> int:
#    "Dati due numeri interi a,b questa funzione fa la somma e restituisce il risultato a+b"
#    return a + b
#
#
# llm = ChatOllama(model="llama3.2:1b")
# agent = create_agent(
#    llm,
#    tools=[summ_numbers],
#    system_prompt="You are a helpful assistant. Be concise and accurate.",
# )
#
#
# result = agent.invoke({"messages": [{"role": "user", "content": "quanto fa 3+3?"}]})
# print(result["messages"][-1].content)


from typing import Literal
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_weather(location: str) -> str:
    "get weather condition in the specified location"
    return f"Sunny in {location}"


def create_agent_demo():
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    agent = create_agent(
        llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Be concise and accurate.",
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Cassino?")]}
    )
    for m in result.get("messages"):
        m.pretty_print()


create_agent_demo()
