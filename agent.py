"""
Builds the tool-calling agent directly with LangGraph — no langchain.agents
wrapper. This is the "modern LangGraph" pattern: a small state graph with
an LLM node and a tool node, looping until the model stops requesting
tools.

(langchain.agents.create_agent, used in the previous version of this file,
is actually built on exactly this pattern under the hood — this version
just writes the graph out explicitly instead of letting create_agent build
it for you. app.py doesn't need to change: both approaches compile down to
a graph with the same .invoke({"messages": [...]}) interface.)
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import search_knowledge_base

load_dotenv()

MODEL_NAME = os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it")

SYSTEM_PROMPT = """You are a helpful assistant for a student-run internal \
knowledge bank about job hunting: interview questions, how internships \
felt, and techniques/skills that helped.

Rules:
- If the question relates to job hunting, interviews, internships, or \
career experiences, ALWAYS call search_knowledge_base first. Do not answer \
from general knowledge alone.
- Base your answer only on what the tool returns.
- Always cite the source filename(s) the information came from, e.g. \
"(source: alice_google_swe_intern.md)".
- If the knowledge base has nothing relevant, say so plainly instead of \
guessing.
- Keep answers concise and practical."""

TOOLS = [search_knowledge_base]


def build_agent():
    """Returns a compiled LangGraph graph. Same interface as before: call
    .invoke({"messages": [...]}) and read result["messages"][-1]."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        # Prepend the system prompt fresh on every call rather than storing
        # it in state, so it doesn't pile up as the conversation grows.
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()
