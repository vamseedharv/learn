from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import sys

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Our tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))


def wikipedia_search(query: str) -> str:
    print("wikipedia_search agent started")
    """Search Wikipedia and return information about a topic."""

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": query,
        "redirects": 1
    }

    response = requests.get(url, params=params)

    data = response.json()

    pages = data["query"]["pages"]

    page = next(iter(pages.values()))

    if "extract" not in page:
        return "Wikipedia article not found."

    return page["extract"][:5000]

def find_employee_details(expression: str) -> str:
    print("Lets work on giving you employee details")
    return "This agent is still in development state"

# Create a chat with the tool
chat = client.chats.create(
    model="gemini-3.1-flash-lite",
    config=types.GenerateContentConfig(
        tools=[calculator, wikipedia_search, find_employee_details]
    )
)


passed_values = sys.argv[1:]
# Talk to the agent
response = chat.send_message(
    passed_values
)

print(response.text)
