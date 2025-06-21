from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
import uvicorn

load_dotenv()

app = FastAPI(title="Mediator Bot API")

# Initialize Cerebras client
cerebras_client = None
api_key = os.getenv("CEREBRAS_API_KEY")
if api_key:
    cerebras_client = Cerebras(api_key=api_key)


class Message(BaseModel):
    role: str
    content: str


class MessageRequest(BaseModel):
    messages: List[Message]


class MessageResponse(BaseModel):
    response: str
    mediation_triggered: bool = False
    observations: Optional[str] = None
    feelings: Optional[str] = None
    needs: Optional[str] = None
    requests: Optional[str] = None


def process_mediation(messages: List[Message]) -> MessageResponse:
    """Process messages for mediation using NVC approach"""
    if not cerebras_client:
        return MessageResponse(
            response="I'm listening. Continue your conversation.",
            mediation_triggered=False,
        )

    # Format messages for the prompt
    messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[-5:]])

    prompt = f"""You are a mediator for a group house chat. I want you to notice messages that are excessively far from being phrased in NVC when conversations are becoming heated. Respond with {{"response": "[[NO MEDIATION NEEDED]]"}} when the convo is ok. Respond as a mediator and with a suggested NVC translation in other cases.
{messages_text}

Think step by step, if mediation is needed return a JSON like this: {{"response": "I notice some tension here. Let me help translate this using NVC.", "observations": "I observe that harsh words were used", "feelings": "There seems to be frustration and hurt", "needs": "The need for respect and understanding", "requests": "Could you try expressing your concern without blame?"}}. Otherwise return {{"response": "[[NO MEDIATION NEEDED]]"}}."""

    try:
        response = cerebras_client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            stream=False,  # Explicitly disable streaming
        )

        ai_response = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            # First check if it's the no mediation needed response
            if "[[NO MEDIATION NEEDED]]" in ai_response:
                return MessageResponse(
                    response="I'm listening. Continue your conversation.",
                    mediation_triggered=False,
                )

            # Try to extract JSON from the response
            json_start = ai_response.find("{")
            json_end = ai_response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                parsed_response = json.loads(json_str)

                return MessageResponse(
                    response=parsed_response.get(
                        "response", "Let me help mediate this conversation."
                    ),
                    mediation_triggered=True,
                    observations=parsed_response.get("observations"),
                    feelings=parsed_response.get("feelings"),
                    needs=parsed_response.get("needs"),
                    requests=parsed_response.get("requests"),
                )
            else:
                # Fallback if no JSON structure found
                return MessageResponse(
                    response=ai_response,
                    mediation_triggered=True,
                )

        except json.JSONDecodeError:
            # Fallback for JSON parsing errors
            return MessageResponse(
                response=ai_response,
                mediation_triggered=True,
            )

    except Exception as e:
        # Fallback for API errors
        return MessageResponse(
            response="Let's take a step back and try to understand each other's perspectives.",
            mediation_triggered=False,
        )


@app.post("/chat")
async def chat(request: MessageRequest) -> MessageResponse:
    """Main chat endpoint with NVC mediation"""
    return process_mediation(request.messages)


@app.get("/")
async def root():
    return {"message": "Mediator Bot API", "ai_enabled": cerebras_client is not None}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)

