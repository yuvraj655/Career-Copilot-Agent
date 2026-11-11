"""Career Copilot Agent: a LangChain tool-calling agent served with LangServe."""
import ast
import operator
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
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

# A text-based ReAct prompt. It does not use Gemini's native function calling,
# so it works with any Gemini model (including ones that need thought signatures).
REACT_TEMPLATE = (
    SYSTEM_PROMPT
    + """

You have access to these tools:

{tools}

Use this exact format:

Question: the input question you must answer
Thought: think about what to do next
Action: the tool to use, which must be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
... (the Thought/Action/Action Input/Observation steps can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

If you do not need a tool, go straight to:
Thought: I can answer directly
Final Answer: your answer

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

prompt = PromptTemplate.from_template(REACT_TEMPLATE)

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-2.5-flash"),
)  # reads GOOGLE_API_KEY from the environment

agent = create_react_agent(llm, TOOLS, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
)


def _to_text(value) -> str:
    """Gemini can return a list of content blocks; turn any shape into plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(value)


def _clean_output(result: dict) -> dict:
    text = _to_text(result.get("output", "")).strip()
    return {"output": text or "Sorry, I could not produce an answer. Please try rephrasing."}


chain = agent_executor | RunnableLambda(_clean_output)


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
    chain.with_types(input_type=AgentInput, output_type=AgentOutput),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
