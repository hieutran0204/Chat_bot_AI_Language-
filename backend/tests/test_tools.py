# name: test_tools.py
# description: Unit tests for BaseTool, DictionaryLookupTool, and GrammarCheckerTool.

import pytest
from app.core.primitives.tools import (
    BaseTool,
    DictionaryLookupTool,
    GrammarCheckerTool,
)


@pytest.mark.asyncio
async def test_dictionary_lookup_tool():
    tool = DictionaryLookupTool()
    assert tool.name == "dictionary_lookup"
    result = await tool.arun(word="resilient")
    assert "resilient" in result
    assert "Definition" in result


@pytest.mark.asyncio
async def test_grammar_checker_tool():
    tool = GrammarCheckerTool()
    assert tool.name == "grammar_checker"
    result = await tool.arun(sentence="She don't like apples.")
    assert "grammar_checker" in tool.name
    assert "She don't like apples." in result
