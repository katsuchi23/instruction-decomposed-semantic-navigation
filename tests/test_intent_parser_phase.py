import importlib
import sys
import types


def _load_intent_parser():
    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv

    if "openai" not in sys.modules:
        openai = types.ModuleType("openai")

        class OpenAI:  # pragma: no cover - parser unit tests do not use the client
            pass

        openai.OpenAI = OpenAI
        sys.modules["openai"] = openai

    if "pydantic" not in sys.modules:
        pydantic = types.ModuleType("pydantic")

        class BaseModel:
            pass

        def Field(*args, **kwargs):
            return None

        pydantic.BaseModel = BaseModel
        pydantic.Field = Field
        sys.modules["pydantic"] = pydantic

    return importlib.import_module("parsing.intent_parser")


def test_extracts_worded_negative_phase_angle():
    parser = _load_intent_parser()

    value = parser._extract_termination_phase_deg(
        "approach the object from negative 45 degree angle"
    )

    assert value == -45.0


def test_extracts_signed_numeric_phase_angle():
    parser = _load_intent_parser()

    value = parser._extract_termination_phase_deg(
        "go to the red cube and stop at angle -45 degrees"
    )

    assert value == -45.0


def test_ignores_reference_side_language_without_termination_cue():
    parser = _load_intent_parser()

    value = parser._extract_termination_phase_deg(
        "go to the red cube near the object on the right side"
    )

    assert value is None
