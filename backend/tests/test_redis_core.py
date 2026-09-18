# name: test_redis_core.py
# description: Unit tests for RedisChatHistory, LearnerProfile, and RedisCacheManager.

import pytest
import unittest.mock as mock
from app.core.primitives.messages import HumanMessage, AIMessage, MessageRole
from app.core.redis.memory import RedisChatMessageHistory
from app.core.redis.profile import LearnerProfile, RedisLearnerProfileManager
from app.core.redis.cache import RedisCacheManager


def test_learner_profile_top_weaknesses():
    profile = LearnerProfile(
        user_id="user-123",
        level="B1",
        weaknesses={"past_tense": 5, "preposition": 8, "articles": 2, "pronunciation_th": 12},
    )
    top = profile.top_weaknesses(top_n=2)
    assert top == [("pronunciation_th", 12), ("preposition", 8)]


@pytest.mark.asyncio
async def test_redis_memory_offline_fallback():
    # If redis client is not connected, methods should not raise exception
    mock_redis = mock.MagicMock()
    mock_redis.is_available = False
    mock_redis.client = None

    mem = RedisChatMessageHistory("conv-1", redis_client=mock_redis)
    await mem.add_message(HumanMessage(content="test"))
    msgs = await mem.get_messages()
    assert msgs == []


@pytest.mark.asyncio
async def test_learner_profile_manager_offline_fallback():
    mock_redis = mock.MagicMock()
    mock_redis.is_available = False
    mock_redis.client = None

    mgr = RedisLearnerProfileManager(redis_client=mock_redis)
    profile = await mgr.get_profile("user-456", fallback_level="A2")
    assert profile.user_id == "user-456"
    assert profile.level == "A2"
    assert profile.weaknesses == {}


def test_cache_convenience_keys():
    dict_key = RedisCacheManager.dict_key("  Apple ")
    assert dict_key == "langai:dict:apple"

    grammar_key = RedisCacheManager.grammar_key("He go to school.")
    assert grammar_key.startswith("langai:grammar:")
