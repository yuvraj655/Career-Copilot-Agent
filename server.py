"""Career Copilot Agent: a LangChain tool-calling agent served with LangServe."""
import ast
import operator
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------- tools ----
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    """Safely evaluate a basic arithmetic expression (no eval())."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic maths expression such as '(8.5 + 7.9) / 2' or '85 * 100 / 120'.
    Use it for any arithmetic, such as CGPA, percentages or salary maths."""
    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"Could not calculate: {exc}"


@tool
def current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time. Optionally pass an IANA timezone such as 'Asia/Kolkata'."""
    try:
        return datetime.now(ZoneInfo(timezone)).strftime("%A, %d %B %Y, %I:%M %p (%Z)")
    except Exception:
        return datetime.now(ZoneInfo("UTC")).strftime("%A, %d %B %Y, %I:%M %p (UTC)")


@tool
def wikipedia_summary(topic: str) -> str:
    """Look up a short factual summary of a technology, company, concept or role on Wikipedia."""
    import wikipedia

    try:
        return wikipedia.summary(topic, sentences=4, auto_suggest=False)
    except wikipedia.DisambiguationError as exc:
        return "That topic is ambiguous. Options: " + ", ".join(exc.options[:5])
    except wikipedia.PageError:
        return "No Wikipedia page found for that topic."
    except Exception as exc:
        return f"Wikipedia lookup failed: {exc}"


TOOLS = [calculator, current_datetime, wikipedia_summary]

# ---------------------------------------------------------------- agent ----
SYSTEM_PROMPT = (
    "You are Career Copilot, a friendly assistant for engineering students in India. "
    "You help with placement preparation, interview practice, resume advice, "
    "learning roadmaps and study planning. Use your tools when they help: "
    "'calculator' for any arithmetic, 'current_datetime' for dates and deadlines, "
    "and 'wikipedia_summary' for factual background. Give clear, practical, "
    "step-by-step answers and keep them concise. If you are unsure, say so."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-2.5-flash"),
    temperature=0.3,
)  # reads GOOGLE_API_KEY from the environment

agent = create_tool_calling_agent(llm, TOOLS, prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)


class AgentInput(BaseModel):
    input: str


class AgentOutput(BaseModel):
    output: str


# ------------------------------------------------------------------ app ----
app = FastAPI(
    title="Career Copilot Agent",
    version="1.0",
    description="A LangChain agent served with LangServe.",
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/agent/playground/")


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


add_routes(
    app,
    agent_executor.with_types(input_type=AgentInput, output_type=AgentOutput),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
