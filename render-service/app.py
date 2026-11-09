"""
Jarvis Alexa Skill - backend, hosted on Render (no AWS, no credit card needed)

Flow:
  You say: "Alexa, ask Jarvis <question>"
  -> Alexa cloud does speech-to-text and POSTs it to this web service over HTTPS
  -> This app calls Groq's API (free tier) with that text
  -> The AI's reply is sent back to Alexa, which speaks it through the Echo Dot

Setup:
  1. Push this folder to a GitHub repo.
  2. On Render.com (free, no card): New -> Web Service -> connect the repo.
     - Build command: pip install -r requirements.txt
     - Start command: gunicorn app:app
  3. In Render's dashboard, add an environment variable:
     - GROQ_API_KEY = your key from console.groq.com/keys
  4. Once deployed, Render gives you a URL like https://jarvis-xxxx.onrender.com
     Your Alexa Skill's endpoint is that URL (Alexa requires HTTPS, which Render
     provides automatically).

You can swap ask_ai() below to call any other AI backend/API instead of Groq.
"""

import os
import logging
import urllib.request
import json as jsonlib

from flask import Flask, request, jsonify
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.serialize import DefaultSerializer
from ask_sdk_model import Response
from ask_sdk_model.request_envelope import RequestEnvelope

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Alexa speech responses are capped; keep replies short so they read back cleanly.
SYSTEM_PROMPT = (
    "You are Jarvis, a concise voice assistant speaking through a smart speaker. "
    "Answer in 1-3 short spoken sentences. No markdown, no lists, no code blocks, "
    "since this will be read aloud."
)


def ask_ai(question: str) -> str:
    """Send the question to Groq (free tier) and return a short spoken-friendly answer."""
    if not GROQ_API_KEY:
        return "I'm not connected to my brain yet. Please set the API key on Render."

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    req = urllib.request.Request(
        GROQ_URL,
        data=jsonlib.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {GROQ_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = jsonlib.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        if not choices:
            return "I didn't get a clear answer back, sorry."
        text = choices[0].get("message", {}).get("content", "")
        return text.strip() or "I didn't get a clear answer back, sorry."
    except Exception:
        logger.exception("AI call failed")
        return "I ran into a problem reaching my AI backend."


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = "Jarvis here. What would you like to know?"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class AskJarvisIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AskJarvisIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        slots = handler_input.request_envelope.request.intent.slots
        query = ""
        if slots and "query" in slots and slots["query"].value:
            query = slots["query"].value

        if not query:
            speak_output = "I didn't catch a question. What would you like to ask?"
            return (
                handler_input.response_builder.speak(speak_output)
                .ask(speak_output)
                .response
            )

        answer = ask_ai(query)
        return (
            handler_input.response_builder.speak(answer)
            .ask("Anything else?")
            .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = "Ask me anything, like 'Alexa, ask Jarvis what time it is in Tokyo.'"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.CancelIntent")(handler_input) or is_intent_name(
            "AMAZON.StopIntent"
        )(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = "Goodbye."
        return handler_input.response_builder.speak(speak_output).response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input: HandlerInput, exception) -> bool:
        return True

    def handle(self, handler_input: HandlerInput, exception) -> Response:
        logger.error(exception, exc_info=True)
        speak_output = "Sorry, something went wrong on my end."
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(AskJarvisIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

skill = sb.create()
serializer = DefaultSerializer()

app = Flask(__name__)

# Optional: set SKILL_ID (your Alexa Skill's ID, from the Alexa Developer Console)
# as a Render environment variable once you have it, to reject requests that
# don't come from your own skill. Safe to leave unset while you're getting it working.
SKILL_ID = os.environ.get("SKILL_ID") or None

# NOTE: This handles raw Alexa JSON directly with ask-sdk-core's skill.invoke(),
# instead of using the flask-ask-sdk / ask-sdk-webservice-support adapter. Those
# packages pull in a dependency (oscrypto, for request-signature verification)
# that fails to load on current Python/OpenSSL versions on Render. Skipping
# official request-signature verification is an acceptable trade-off for a
# private/development-mode skill; set SKILL_ID above for a basic sanity check.
@app.route("/", methods=["POST"])
def invoke_skill():
    body = request.get_json(force=True, silent=True) or {}

    if SKILL_ID:
        sent_id = (
            body.get("context", {})
            .get("System", {})
            .get("application", {})
            .get("applicationId")
        )
        if sent_id and sent_id != SKILL_ID:
            return jsonify({"error": "invalid applicationId"}), 400

    request_envelope = serializer.deserialize(body, RequestEnvelope)
    response_envelope = skill.invoke(request_envelope=request_envelope, context=None)
    response_dict = serializer.serialize(response_envelope)
    return jsonify(response_dict)


@app.route("/health")
def health():
    return {"status": "ok"}, 200
