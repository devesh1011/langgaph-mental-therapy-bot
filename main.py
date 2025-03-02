from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import uuid
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from llm import llm
from fastapi.middleware.cors import CORSMiddleware


# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize LangGraph
def chatbot(state: State):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a friendly and relatable mental health assistant, acting as a peer therapist for young adults aged 18-27. Engage in casual, empathetic conversations, using language and references that resonate with Gen Z.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt_template.invoke({"messages": state["messages"]})
    response = llm.invoke(prompt)
    return {"messages": response}


builder = StateGraph(State)
builder.add_node("model", chatbot)
builder.add_edge(START, "model")
builder.add_edge("model", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.post("/new-chat")
async def new_chat():
    """
    Creates a new chat thread and returns the thread_id.
    """
    # Generate a unique thread_id
    thread_id = str(uuid.uuid4())

    # Initialize an empty conversation state for the thread
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": []}  # Empty messages list for a new chat
    graph.stream(input_data, config)  # Initialize the thread in memory

    # Return the thread_id to the frontend
    return {"thread_id": thread_id}


@app.post("/chat")
async def chat(request: dict):
    message = request.get("message")
    thread_id = request.get("thread_id")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    # Generate a new thread_id if not provided
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Prepare the state
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=message)]}

    # Stream the response
    def stream_response():
        try:
            for chunk, _ in graph.stream(input_data, config, stream_mode="messages"):
                if isinstance(chunk, AIMessage):
                    yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_response(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
