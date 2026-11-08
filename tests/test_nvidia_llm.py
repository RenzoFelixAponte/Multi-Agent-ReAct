import io
import json
import os
import urllib.error

import pytest

from adapters import nvidia_llm
from adapters.dotenv_loader import load_dotenv
from adapters.nvidia_llm import NvidiaLLM, parse_completion


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(nvidia_llm, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        NvidiaLLM()


def test_load_dotenv_reads_file_without_overriding_env(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('NVIDIA_API_KEY="clave-del-archivo"\n# comentario\nNVIDIA_MODEL=mi-modelo\n', encoding="utf-8")
    monkeypatch.setenv("NVIDIA_MODEL", "ya-existe")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    load_dotenv(env)
    assert os.environ["NVIDIA_API_KEY"] == "clave-del-archivo"
    assert os.environ["NVIDIA_MODEL"] == "ya-existe"


def test_complete_sends_bearer_and_model_and_returns_text(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        payload = {"choices": [{"message": {"content": "  Lima  "}}]}
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(nvidia_llm.urllib.request, "urlopen", fake_urlopen)
    llm = NvidiaLLM(api_key="k-test", model="nvidia/modelo-x")
    assert llm.complete("¿Capital del Perú?") == "Lima"
    assert captured["auth"] == "Bearer k-test"
    assert captured["body"]["model"] == "nvidia/modelo-x"
    assert captured["body"]["messages"] == [{"role": "user", "content": "¿Capital del Perú?"}]


def test_http_error_is_wrapped_with_status(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(nvidia_llm.urllib.request, "urlopen", fake_urlopen)
    llm = NvidiaLLM(api_key="mala", model="m")
    with pytest.raises(RuntimeError, match="401"):
        llm.complete("hola")


def test_parse_completion_handles_empty_content():
    assert parse_completion({"choices": [{"message": {"content": None}}]}) == ""
