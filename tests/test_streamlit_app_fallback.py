from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit_app import build_demo_response


def test_build_demo_response_contains_demo_mode_and_prompt():
    response = build_demo_response("Hello there")
    assert "Demo mode" in response
    assert "Hello there" in response
