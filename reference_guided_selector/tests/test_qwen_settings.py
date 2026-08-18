import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_settings_select_local_vllm_model():
    settings = json.loads((ROOT / ".qwen" / "settings.json").read_text())

    assert settings["security"]["auth"]["selectedType"] == "openai"
    assert settings["model"]["name"] == "qwen3.8-27b"

    provider = settings["modelProviders"]["openai"][0]
    assert provider["envKey"] == "VLLM_API_KEY"
    assert provider["baseUrl"] == "http://127.0.0.1:8000/v1"
    assert provider["generationConfig"]["timeout"] == 300000
    assert provider["generationConfig"]["contextWindowSize"] == 65536
    assert provider["generationConfig"]["samplingParams"] == {
        "temperature": 0.2,
        "max_tokens": 8192,
    }
