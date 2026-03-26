"""Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable.
Skip automatically when the key is not available.
"""

import importlib.util
import os
import sys
import unittest

# Load llm_provider directly from file to avoid iGPT.__init__ model imports
_mod_path = os.path.join(
    os.path.dirname(__file__), "..", "iGPT", "controllers", "llm_provider.py"
)
_spec = importlib.util.spec_from_file_location("llm_provider", _mod_path)
llm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_mod)

MiniMaxLLM = llm_mod.MiniMaxLLM
create_llm = llm_mod.create_llm

# Skip the entire module if no API key is available
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Live integration tests against the MiniMax API."""

    def test_basic_completion(self):
        llm = MiniMaxLLM(api_key=MINIMAX_API_KEY, temperature=0.01)
        result = llm._call("Say hello in one word.")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_highspeed_model(self):
        llm = MiniMaxLLM(
            api_key=MINIMAX_API_KEY,
            model_name="MiniMax-M2.7-highspeed",
            temperature=0.01,
        )
        result = llm._call("What is 2+2? Answer with just the number.")
        self.assertIsInstance(result, str)
        self.assertIn("4", result)

    def test_create_llm_factory(self):
        llm = create_llm(provider="minimax", api_key=MINIMAX_API_KEY)
        self.assertIsInstance(llm, MiniMaxLLM)
        result = llm._call("Reply with the word 'pong'.")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()
