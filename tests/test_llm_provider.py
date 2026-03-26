"""Unit tests for the InternGPT LLM provider module."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Load llm_provider directly from file to avoid iGPT.__init__ model imports
_mod_path = os.path.join(
    os.path.dirname(__file__), "..", "iGPT", "controllers", "llm_provider.py"
)
_spec = importlib.util.spec_from_file_location("llm_provider", _mod_path)
llm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_mod)

MiniMaxLLM = llm_mod.MiniMaxLLM
create_llm = llm_mod.create_llm
detect_provider = llm_mod.detect_provider
MINIMAX_API_BASE = llm_mod.MINIMAX_API_BASE
MINIMAX_DEFAULT_MODEL = llm_mod.MINIMAX_DEFAULT_MODEL
MINIMAX_MODELS = llm_mod.MINIMAX_MODELS
SUPPORTED_PROVIDERS = llm_mod.SUPPORTED_PROVIDERS
_chat_completion = llm_mod._chat_completion


class TestMiniMaxLLM(unittest.TestCase):
    """Tests for the MiniMaxLLM class."""

    def test_default_parameters(self):
        llm = MiniMaxLLM(api_key="test-key")
        self.assertEqual(llm.model_name, MINIMAX_DEFAULT_MODEL)
        self.assertEqual(llm.api_base, MINIMAX_API_BASE)
        self.assertEqual(llm.api_key, "test-key")
        self.assertEqual(llm._llm_type, "minimax")

    def test_temperature_clamping_zero(self):
        llm = MiniMaxLLM(api_key="test-key", temperature=0)
        self.assertEqual(llm.temperature, 0.01)

    def test_temperature_clamping_negative(self):
        llm = MiniMaxLLM(api_key="test-key", temperature=-1)
        self.assertEqual(llm.temperature, 0.01)

    def test_temperature_valid_preserved(self):
        llm = MiniMaxLLM(api_key="test-key", temperature=0.5)
        self.assertEqual(llm.temperature, 0.5)

    def test_temperature_one_preserved(self):
        llm = MiniMaxLLM(api_key="test-key", temperature=1.0)
        self.assertEqual(llm.temperature, 1.0)

    def test_custom_model_name(self):
        llm = MiniMaxLLM(api_key="key", model_name="MiniMax-M2.7-highspeed")
        self.assertEqual(llm.model_name, "MiniMax-M2.7-highspeed")

    def test_custom_api_base(self):
        llm = MiniMaxLLM(api_key="key", api_base="https://custom.api/v1")
        self.assertEqual(llm.api_base, "https://custom.api/v1")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=False)
    def test_api_key_from_env(self):
        llm = MiniMaxLLM()
        self.assertEqual(llm.api_key, "env-key")

    def test_identifying_params(self):
        llm = MiniMaxLLM(api_key="key")
        params = llm._identifying_params
        self.assertIn("model_name", params)
        self.assertIn("temperature", params)
        self.assertIn("api_base", params)

    def test_strip_think_tags_basic(self):
        llm = MiniMaxLLM(api_key="key")
        text = "<think>internal reasoning</think>Hello world"
        self.assertEqual(llm._strip_think_tags(text), "Hello world")

    def test_strip_think_tags_multiline(self):
        llm = MiniMaxLLM(api_key="key")
        text = "<think>\nstep 1\nstep 2\n</think>\nResult: 42"
        self.assertEqual(llm._strip_think_tags(text), "Result: 42")

    def test_strip_think_tags_no_tags(self):
        llm = MiniMaxLLM(api_key="key")
        text = "No thinking here"
        self.assertEqual(llm._strip_think_tags(text), "No thinking here")

    def test_strip_think_tags_multiple(self):
        llm = MiniMaxLLM(api_key="key")
        text = "<think>a</think>Hello <think>b</think>World"
        self.assertEqual(llm._strip_think_tags(text), "Hello World")

    @patch.object(llm_mod, "_chat_completion")
    def test_call_invokes_chat_completion(self, mock_cc):
        mock_cc.return_value = "response text"
        llm = MiniMaxLLM(api_key="test-key", model_name="MiniMax-M2.7")
        result = llm._call("Hello")

        mock_cc.assert_called_once_with(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.01,
            stop=None,
            api_key="test-key",
            api_base=MINIMAX_API_BASE,
        )
        self.assertEqual(result, "response text")

    @patch.object(llm_mod, "_chat_completion")
    def test_call_strips_think_tags_from_response(self, mock_cc):
        mock_cc.return_value = "<think>reasoning</think>Final answer"
        llm = MiniMaxLLM(api_key="test-key")
        result = llm._call("Question")
        self.assertEqual(result, "Final answer")

    @patch.object(llm_mod, "_chat_completion")
    def test_call_with_stop_sequences(self, mock_cc):
        mock_cc.return_value = "answer"
        llm = MiniMaxLLM(api_key="test-key")
        llm._call("prompt", stop=["\n"])
        call_kwargs = mock_cc.call_args[1]
        self.assertEqual(call_kwargs["stop"], ["\n"])

    @patch.object(llm_mod, "_chat_completion")
    def test_call_handles_empty_response(self, mock_cc):
        mock_cc.return_value = ""
        llm = MiniMaxLLM(api_key="test-key")
        result = llm._call("prompt")
        self.assertEqual(result, "")


class TestDetectProvider(unittest.TestCase):
    """Tests for the detect_provider function."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=False)
    def test_explicit_minimax(self):
        self.assertEqual(detect_provider(), "minimax")

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=False)
    def test_explicit_openai(self):
        self.assertEqual(detect_provider(), "openai")

    @patch.dict(os.environ, {"LLM_PROVIDER": "MINIMAX"}, clear=False)
    def test_case_insensitive(self):
        self.assertEqual(detect_provider(), "minimax")

    def test_auto_detect_minimax(self):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("LLM_PROVIDER", None)
        env["MINIMAX_API_KEY"] = "key123"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(detect_provider(), "minimax")

    def test_default_openai(self):
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        env.pop("LLM_PROVIDER", None)
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(detect_provider(), "openai")

    def test_both_keys_defaults_openai(self):
        env = os.environ.copy()
        env.pop("LLM_PROVIDER", None)
        env["MINIMAX_API_KEY"] = "mm_key"
        env["OPENAI_API_KEY"] = "oai_key"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(detect_provider(), "openai")


class TestCreateLLM(unittest.TestCase):
    """Tests for the create_llm factory function."""

    def test_create_minimax_explicit(self):
        llm = create_llm(provider="minimax", api_key="test-key")
        self.assertIsInstance(llm, MiniMaxLLM)
        self.assertEqual(llm.api_key, "test-key")

    def test_create_minimax_with_model(self):
        llm = create_llm(
            provider="minimax",
            api_key="key",
            model_name="MiniMax-M2.7-highspeed",
        )
        self.assertIsInstance(llm, MiniMaxLLM)
        self.assertEqual(llm.model_name, "MiniMax-M2.7-highspeed")

    def test_create_minimax_temperature(self):
        llm = create_llm(provider="minimax", api_key="key", temperature=0.8)
        self.assertEqual(llm.temperature, 0.8)

    def test_create_minimax_zero_temperature_clamped(self):
        llm = create_llm(provider="minimax", api_key="key", temperature=0)
        self.assertEqual(llm.temperature, 0.01)

    @patch.object(llm_mod, "detect_provider", return_value="minimax")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=False)
    def test_auto_detect_minimax(self, mock_detect):
        llm = create_llm()
        self.assertIsInstance(llm, MiniMaxLLM)

    def test_provider_case_insensitive(self):
        llm = create_llm(provider="MiniMax", api_key="key")
        self.assertIsInstance(llm, MiniMaxLLM)


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_minimax_api_base(self):
        self.assertEqual(MINIMAX_API_BASE, "https://api.minimax.io/v1")

    def test_minimax_default_model(self):
        self.assertEqual(MINIMAX_DEFAULT_MODEL, "MiniMax-M2.7")

    def test_minimax_models_list(self):
        self.assertIn("MiniMax-M2.7", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.7-highspeed", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5-highspeed", MINIMAX_MODELS)

    def test_supported_providers(self):
        self.assertIn("openai", SUPPORTED_PROVIDERS)
        self.assertIn("minimax", SUPPORTED_PROVIDERS)


if __name__ == "__main__":
    unittest.main()
