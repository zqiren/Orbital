"""Tests for token budget derivation from dollar budget."""
import pytest
from agent_os.agent.pricing import budget_usd_to_token_budget, get_cost_rates
from agent_os.daemon_v2.models import AgentConfig


class TestBudgetUsdToTokenBudget:
    """budget_usd_to_token_budget converts dollars to token budget."""

    def test_basic_conversion_sonnet(self):
        """$5 with Sonnet pricing ($3/$15 per 1M) -> ~1.04M tokens."""
        # cost_per_1k: input=0.003, output=0.015
        result = budget_usd_to_token_budget(5.0, 0.003, 0.015)
        # blended = 0.85*0.003 + 0.15*0.015 = 0.00255 + 0.00225 = 0.0048 per 1K
        # token_budget = 5.0 / 0.0048 * 1000 = 1_041_666
        assert 1_000_000 < result < 1_100_000

    def test_basic_conversion_cheap_model(self):
        """$5 with cheap model ($0.25/$1.25 per 1M) -> ~12.5M tokens."""
        # cost_per_1k: input=0.00025, output=0.00125
        result = budget_usd_to_token_budget(5.0, 0.00025, 0.00125)
        assert 10_000_000 < result < 15_000_000

    def test_basic_conversion_expensive_model(self):
        """$5 with expensive model ($15/$75 per 1M) -> ~232K tokens."""
        # cost_per_1k: input=0.015, output=0.075
        result = budget_usd_to_token_budget(5.0, 0.015, 0.075)
        assert 200_000 < result < 300_000

    def test_zero_budget_returns_zero(self):
        """$0 budget -> 0 token budget."""
        result = budget_usd_to_token_budget(0.0, 0.003, 0.015)
        assert result == 0

    def test_zero_cost_rates_returns_safety_fallback(self):
        """Zero cost rates -> high safety fallback."""
        result = budget_usd_to_token_budget(5.0, 0.0, 0.0)
        assert result == 100_000_000

    def test_none_budget_returns_safety_fallback(self):
        """None budget -> high safety fallback (100M)."""
        result = budget_usd_to_token_budget(None, 0.003, 0.015)
        assert result == 100_000_000

    def test_returns_int(self):
        """Result is always an integer."""
        result = budget_usd_to_token_budget(5.0, 0.003, 0.015)
        assert isinstance(result, int)


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
