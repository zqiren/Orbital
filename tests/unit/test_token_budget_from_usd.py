"""Tests for the PURE token-budget safety-net cap.

Budget Piece 2 DELETED ``budget_usd_to_token_budget`` (the dollar→token
derivation that fed the loop's token gate as a second budget-blocking path).
The loop's token gate now serves only as a NON-budget safety net, fed directly
by ``AgentConfig.token_budget`` (default 100M). These tests pin that pure cap
and the per-1K pricing lookup; the dollar-derivation tests were removed with the
function.
"""
import pytest
from agent_os.agent.pricing import get_cost_rates
from agent_os.daemon_v2.models import AgentConfig


class TestAgentConfigTokenBudgetDefault:
    """AgentConfig default token_budget is 100M (safety net)."""

    def test_default_is_safety_net(self):
        config = AgentConfig(workspace="/tmp", model="m", api_key="k")
        assert config.token_budget == 100_000_000

    def test_custom_token_budget_preserved(self):
        config = AgentConfig(workspace="/tmp", model="m", api_key="k", token_budget=50_000)
        assert config.token_budget == 50_000


class TestDeepSeekPricingFromProvidersJson:
    """get_cost_rates loads DeepSeek prices from the real providers.json.

    deepseek-v4-pro was overstated 4x (1.74/3.48) vs the provider's
    published USD per-1M rates of 0.435/0.87
    (https://api-docs.deepseek.com/quick_start/pricing, fetched 2026-06-08).
    These assert the corrected number loads, and anchor the per-1M -> per-1K
    conversion in get_cost_rates.
    """

    def test_deepseek_v4_pro_corrected_rate(self):
        # providers.json stores 0.435/0.87 per 1M; get_cost_rates returns per-1K.
        ci, co = get_cost_rates("deepseek-v4-pro", "deepseek")
        assert ci == pytest.approx(0.435 / 1000)
        assert co == pytest.approx(0.87 / 1000)

    def test_deepseek_v4_flash_unchanged_rate(self):
        # Verified correct against the same source; guards the conversion path.
        ci, co = get_cost_rates("deepseek-v4-flash", "deepseek")
        assert ci == pytest.approx(0.14 / 1000)
        assert co == pytest.approx(0.28 / 1000)
