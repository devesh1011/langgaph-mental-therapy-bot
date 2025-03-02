from langgraph.graph import StateGraph, START, END
from llm import llm
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import uuid
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    SystemMessage,
    trim_messages,
    AIMessage,
    HumanMessage,
)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a friendly and relatable mental health assistant, acting as a peer therapist for young adults aged 18-27. Engage in casual, empathetic conversations, using language and references that resonate with Gen Z. Provide support on topics like stress, relationships, and self-discovery, ensuring your responses are approachable, non-judgmental, and infused with a sense of camaraderie.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def user_message(state: State):
    # Extract the latest user message from the state's messages
    user_msg = state["messages"][
        -1
    ].content  # Accessing the 'content' attribute directly
    return {"messages": add_messages(user_msg)}


def chatbot(state: State):
    # trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {
            "messages": state["messages"],
        }
    )
    response = llm.invoke(prompt)
    return {"messages": response}


builder = StateGraph(State)
# builder.add_node("user_msg", user_message)
# builder.add_node("chatbot", chatbot)

# builder.add_edge(START, "user_msg")
# builder.add_edge("user_msg", "chatbot")
# builder.add_edge("chatbot", END)
builder.add_edge(START, "model")
builder.add_node("model", chatbot)


checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": uuid.uuid4()}}

query = "What's your job?"

# input_messages = [HumanMessage(query)]
for chunk, metadata in graph.stream(
    {"messages": query},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="")
