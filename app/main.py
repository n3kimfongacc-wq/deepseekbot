import os
import json
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Enable CORS (for testing — restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepSeek client (OpenAI-compatible)
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# --- Tool definition ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_drink_price",
            "description": "Get the price of a drink based on item, size, ice, and takeaway options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "enum": ["Kopi", "Kopi O", "Teh", "Teh O", "Milo", "Nescafe"],
                        "description": "The name of the drink"
                    },
                    "size": {
                        "type": "string",
                        "enum": ["Normal", "Big"],
                        "default": "Normal"
                    },
                    "ice": {"type": "boolean", "default": False},
                    "dabao": {"type": "boolean", "default": False}
                },
                "required": ["item"]
            }
        }
    }
]

# --- Menu data (from your Excel) ---
# Format: (Normal, Big, Ais, Dabao, Cup)
DRINK_PRICES = {
    "Kopi":    (2.3, 2.8, 2.8, 3.2, 3.2),
    "Kopi O":  (2.3, 2.8, 2.8, 3.2, 3.2),
    "Teh":     (2.3, 2.8, 2.8, 3.2, 3.2),
    "Teh O":   (2.3, 2.8, 2.8, 3.2, 3.2),
    "Milo":    (2.8, 3.8, 3.5, 4.2, 4.2),
    "Nescafe": (2.8, 3.8, 3.8, 4.2, 4.2),
}

def execute_drink_price(item: str, size: str = "Normal", ice: bool = False, dabao: bool = False):
    """Backend function that returns the actual price."""
    prices = DRINK_PRICES.get(item)
    if not prices:
        return {"error": f"Item '{item}' not found."}

    normal, big, ais, dabao_price, cup = prices

    if dabao:
        return {"item": item, "price": dabao_price, "condition": "Dabao"}
    if ice:
        return {"item": item, "price": ais if size == "Normal" else big, "condition": "Ais"}
    if size == "Big":
        return {"item": item, "price": big, "condition": "Big"}
    return {"item": item, "price": normal, "condition": "Normal"}

# --- Request model ---
class ChatRequest(BaseModel):
    message: str

# --- Chat endpoint ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = [{"role": "user", "content": req.message}]

    # Round 1: model decides whether to call a tool
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    message = response.choices[0].message

    # Handle tool call
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if function_name == "get_drink_price":
            result = execute_drink_price(**args)
        else:
            result = {"error": "Unknown function"}

        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

        # Round 2: model generates friendly reply
        final_response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tools
        )
        final_message = final_response.choices[0].message

        return {
            "reply": final_message.content,
            "tool_used": function_name,
            "tool_result": result
        }

    return {"reply": message.content}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Serve the chat UI (must be LAST)
app.mount("/", StaticFiles(directory="public", html=True), name="static")
