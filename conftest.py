"""
conftest.py
-----------
Pytest configuration for the P54 project.
Placed at the project root so pytest can find all modules
regardless of which directory tests are run from.
"""

import sys
import os

# Add project root to path so all absolute imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

collect_ignore = ["llm_backend/LLM_eval/test_cases.py"]