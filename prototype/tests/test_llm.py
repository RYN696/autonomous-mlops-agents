import pytest
import requests

from common.llm import LLMError, OllamaClient


class FakeResponse:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return {"message": {"content": self.content}}


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.sent = response, error, None

    def post(self, url, json=None, timeout=None):
        self.sent = (url, json)
        if self.error:
            raise self.error
        return self.response


def test_ollama_returns_parsed_json_with_deterministic_settings():
    session = FakeSession(FakeResponse('{"fin": true}'))
    client = OllamaClient("qwen2.5:3b", url="http://ollama:11434", session=session)
    assert client.chat_json("système", "utilisateur") == {"fin": True}
    url, payload = session.sent
    assert url == "http://ollama:11434/api/chat"
    assert payload["format"] == "json" and payload["options"]["temperature"] == 0
    assert payload["keep_alive"]                       # modèle gardé en mémoire GPU


def test_a_json_schema_is_sent_as_the_format_instead_of_the_plain_json_mode():
    schema = {"type": "object", "properties": {"incident": {"type": "boolean"}}, "required": ["incident"]}
    session = FakeSession(FakeResponse('{"incident": false}'))
    client = OllamaClient("qwen3:4b", session=session)
    assert client.chat([{"role": "user", "content": "u"}], schema=schema) == {"incident": False}
    assert session.sent[1]["format"] == schema          # "json" seul laisse le modèle recopier le contexte


@pytest.mark.parametrize("session", [
    FakeSession(FakeResponse("pas du json")),
    FakeSession(FakeResponse('["une", "liste"]')),
    FakeSession(FakeResponse("{}", status=500)),
    FakeSession(error=requests.ConnectionError("Ollama arrêté")),
])
def test_every_failure_becomes_an_llm_error(session):
    with pytest.raises(LLMError):
        OllamaClient("m", session=session).chat_json("s", "u")


def test_thinking_mode_is_turned_off_for_qwen3_but_not_sent_to_other_models(monkeypatch):
    from common import config
    monkeypatch.setattr(config, "LLM_THINK", "off")
    qwen3 = FakeSession(FakeResponse('{"ok": true}'))
    OllamaClient("qwen3:4b", session=qwen3).chat_json("s", "u")
    assert qwen3.sent[1]["think"] is False             # sinon qwen3 "réfléchit" longuement : trop lent
    qwen25 = FakeSession(FakeResponse('{"ok": true}'))
    OllamaClient("qwen2.5:3b", session=qwen25).chat_json("s", "u")
    assert "think" not in qwen25.sent[1]
    monkeypatch.setattr(config, "LLM_THINK", "auto")
    auto = FakeSession(FakeResponse('{"ok": true}'))
    OllamaClient("qwen3:4b", session=auto).chat_json("s", "u")
    assert "think" not in auto.sent[1]
