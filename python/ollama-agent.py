from ollama import chat
from dotenv import load_dotenv
import requests
import sys

load_dotenv()


# ============================================================
# TOOL 1: Calculator
# ============================================================

def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        # NOTE: eval() is unsafe for untrusted input.
        # This is kept here only to match your original example.
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating expression: {e}"


# ============================================================
# TOOL 2: Wikipedia Search
# ============================================================

def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return information about a topic."""

    print("\n[wikipedia_search tool called]")

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": query,
        "redirects": 1
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        pages = data["query"]["pages"]
        page = next(iter(pages.values()))

        if "extract" not in page:
            return "Wikipedia article not found."

        return page["extract"][:5000]

    except Exception as e:
        return f"Wikipedia search failed: {e}"


# ============================================================
# TOOL 3: Employee Details
# ============================================================

def find_employee_details(expression: str) -> str:
    """Find employee details using an employee ID or search expression."""

    print("\n[find_employee_details tool called]")

    return "This employee details service is still under development."


# ============================================================
# TOOL DEFINITIONS
# ============================================================

tools = [
    calculator,
    wikipedia_search,
    find_employee_details
]


# ============================================================
# GET USER INPUT
# ============================================================

passed_values = sys.argv[1:]

if not passed_values:
    print("Please provide a question.")
    print("Example:")
    print('python agent.py "What is 25 * 30?"')
    sys.exit(1)

user_message = " ".join(passed_values)


# ============================================================
# INITIAL MESSAGE
# ============================================================

messages = [
    {
        "role": "system",
        "content": """
You are a helpful AI assistant.

You have access to several tools.

Use the calculator tool for mathematical calculations.

Use the wikipedia_search tool when the user asks for factual
information that should be searched on Wikipedia.

Use the find_employee_details tool when the user asks for
employee information.

Use tools when appropriate rather than guessing.
"""
    },
    {
        "role": "user",
        "content": user_message
    }
]


# ============================================================
# AGENT LOOP
# ============================================================

while True:

    response = chat(
        model="qwen3",
        messages=messages,
        tools=tools
    )

    assistant_message = response.message

    # Add Ollama's response to conversation history
    messages.append(assistant_message)

    # --------------------------------------------------------
    # Check whether the model wants to call a tool
    # --------------------------------------------------------

    if not assistant_message.tool_calls:
        break

    # --------------------------------------------------------
    # Execute requested tools
    # --------------------------------------------------------

    for tool_call in assistant_message.tool_calls:

        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments

        print(f"\n[Tool requested: {tool_name}]")
        print(f"[Arguments: {tool_arguments}]")

        if tool_name == "calculator":

            result = calculator(
                tool_arguments["expression"]
            )

        elif tool_name == "wikipedia_search":

            result = wikipedia_search(
                tool_arguments["query"]
            )

        elif tool_name == "find_employee_details":

            result = find_employee_details(
                tool_arguments["expression"]
            )

        else:

            result = f"Unknown tool: {tool_name}"

        # Send tool result back to Ollama
        messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": result
        })


# ============================================================
# FINAL ANSWER
# ============================================================

print("\nAssistant:")
print(assistant_message.content)