# Token Budget Derived from Dollar Budget — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded 500K token budget with one derived from the user's dollar budget using per-model pricing from providers.json.

**Architecture:** Add `budget_usd_to_token_budget()` to `pricing.py` that converts a dollar amount to a cumulative token budget using a blended input/output cost rate. Wire it through `agent_manager.py` so both AgentConfig construction paths use it. Raise the default token_budget in `models.py` to act as a safety net only.

**Tech Stack:** Python, pytest, existing pricing infrastructure

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `agent_os/agent/pricing.py` | Modify | Add `budget_usd_to_token_budget()` conversion function |
| `agent_os/daemon_v2/models.py` | Modify | Raise default `token_budget` from 500K to 100M |
| `agent_os/daemon_v2/agent_manager.py` | Modify | Derive `token_budget` from `budget_limit_usd` in both config paths |
| `tests/unit/test_token_budget_from_usd.py` | Create | Unit tests for the conversion + wiring |

---

### Task 1: Add `budget_usd_to_token_budget()` to pricing.py

**Files:**
- Modify: `agent_os/agent/pricing.py`
- Create: `tests/unit/test_token_budget_from_usd.py`

- [ ] **Step 1: Write failing tests for the conversion function**

Create `tests/unit/test_token_budget_from_usd.py`:

```python
"""Tests for token budget derivation from dollar budget."""
import pytest
from agent_os.agent.pricing import budget_usd_to_token_budget


class TestBudgetUsdToTokenBudget:
    """budget_usd_to_token_budget converts dollars to token budget."""

    def test_basic_conversion_sonnet(self):
        """$5 with Sonnet pricing ($3/$15 per 1M) → ~1.19M tokens."""
        # cost_per_1k: input=0.003, output=0.015
        result = budget_usd_to_token_budget(5.0, 0.003, 0.015)
        # blended = 0.85*0.003 + 0.15*0.015 = 0.00255 + 0.00225 = 0.0048 per 1K
        # token_budget = 5.0 / 0.0048 * 1000 = 1_041_666
        assert 1_000_000 < result < 1_100_000

    def test_basic_conversion_cheap_model(self):
        """$5 with cheap model ($0.25/$1.25 per 1M) → ~12.5M tokens."""
        # cost_per_1k: input=0.00025, output=0.00125
        result = budget_usd_to_token_budget(5.0, 0.00025, 0.00125)
        assert 10_000_000 < result < 15_000_000

    def test_basic_conversion_expensive_model(self):
        """$5 with expensive model ($15/$75 per 1M) → ~232K tokens."""
        # cost_per_1k: input=0.015, output=0.075
        result = budget_usd_to_token_budget(5.0, 0.015, 0.075)
        assert 200_000 < result < 300_000

    def test_zero_budget_returns_zero(self):
        """$0 budget → 0 token budget."""
        result = budget_usd_to_token_budget(0.0, 0.003, 0.015)
        assert result == 0

    def test_zero_cost_rates_returns_safety_fallback(self):
        """Zero cost rates → high safety fallback."""
        result = budget_usd_to_token_budget(5.0, 0.0, 0.0)
        assert result == 100_000_000

    def test_none_budget_returns_safety_fallback(self):
        """None budget → high safety fallback (100M)."""
        result = budget_usd_to_token_budget(None, 0.003, 0.015)
        assert result == 100_000_000

    def test_returns_int(self):
        """Result is always an integer."""
        result = budget_usd_to_token_budget(5.0, 0.003, 0.015)
        assert isinstance(result, int)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_token_budget_from_usd.py -v`
Expected: FAIL with "cannot import name 'budget_usd_to_token_budget'"

- [ ] **Step 3: Implement `budget_usd_to_token_budget()` in pricing.py**

Add to the end of `agent_os/agent/pricing.py`:

```python
# Default safety-net token budget when no dollar budget is configured
_SAFETY_NET_TOKEN_BUDGET = 100_000_000  # 100M tokens

# Assumed ratio of input to output tokens in agentic workloads.
# Input dominates because the growing conversation context is re-sent each turn.
_INPUT_TOKEN_RATIO = 0.85


def budget_usd_to_token_budget(
    budget_usd: float | None,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    input_ratio: float = _INPUT_TOKEN_RATIO,
) -> int:
    """Convert a dollar budget to an approximate cumulative token budget.

    Uses a blended cost rate weighted by input_ratio (default 85% input,
    15% output) to account for agentic workloads where input tokens dominate
    due to context re-sending.

    Returns _SAFETY_NET_TOKEN_BUDGET when budget_usd is None (no cap set).
    """
    if budget_usd is None:
        return _SAFETY_NET_TOKEN_BUDGET

    if budget_usd <= 0:
        return 0

    blended_per_1k = input_ratio * cost_per_1k_input + (1 - input_ratio) * cost_per_1k_output
    if blended_per_1k <= 0:
        return _SAFETY_NET_TOKEN_BUDGET

    return int(budget_usd / blended_per_1k * 1000)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_token_budget_from_usd.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_os/agent/pricing.py tests/unit/test_token_budget_from_usd.py
git commit -m "feat: add budget_usd_to_token_budget() conversion function"
```

---

### Task 2: Raise default token_budget in models.py

**Files:**
- Modify: `agent_os/daemon_v2/models.py:71`

- [ ] **Step 1: Update default token_budget**

In `agent_os/daemon_v2/models.py` line 71, change:

```python
# Old:
token_budget: int = 500_000
# New:
token_budget: int = 100_000_000
```

- [ ] **Step 2: Update tests that reference the old default**

In `tests/platform/test_consumer1_wiring.py` lines 322-324, update:
```python
def test_agent_config_has_token_budget(self):
    """AgentConfig has token_budget with default 100_000_000."""
    ...
    assert config.token_budget == 100_000_000
```

In `tests/unit/test_component_f.py` line 71, update:
```python
assert cfg.token_budget == 100_000_000
```

- [ ] **Step 3: Run affected tests**

Run: `python -m pytest tests/platform/test_consumer1_wiring.py tests/unit/test_component_f.py -v -k token_budget`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add agent_os/daemon_v2/models.py tests/platform/test_consumer1_wiring.py tests/unit/test_component_f.py
git commit -m "feat: raise default token_budget to 100M (safety net only)"
```

---

### Task 3: Wire token_budget derivation in agent_manager.py

**Files:**
- Modify: `agent_os/daemon_v2/agent_manager.py:530-549` (start_agent path)
- Modify: `agent_os/daemon_v2/agent_manager.py:767-780` (inject_message path)
- Extend: `tests/unit/test_token_budget_from_usd.py`

- [ ] **Step 1: Write wiring test**

Add to `tests/unit/test_token_budget_from_usd.py`:

```python
from agent_os.daemon_v2.models import AgentConfig


class TestAgentConfigTokenBudgetDefault:
    """AgentConfig default token_budget is 100M (safety net)."""

    def test_default_is_safety_net(self):
        config = AgentConfig(workspace="/tmp", model="m", api_key="k")
        assert config.token_budget == 100_000_000

    def test_custom_token_budget_preserved(self):
        config = AgentConfig(workspace="/tmp", model="m", api_key="k", token_budget=50_000)
        assert config.token_budget == 50_000
```

- [ ] **Step 2: Modify start_agent path in agent_manager.py**

In `agent_os/daemon_v2/agent_manager.py`, after line 533 (where `cost_per_1k_input/output` are obtained), add the token_budget derivation before the AgentLoop construction:

```python
        from agent_os.agent.pricing import get_cost_rates, budget_usd_to_token_budget
        cost_per_1k_input, cost_per_1k_output = get_cost_rates(
            config.model, config.provider,
        )

        # Derive token budget from dollar budget when set
        effective_token_budget = budget_usd_to_token_budget(
            config.budget_limit_usd, cost_per_1k_input, cost_per_1k_output,
        )
```

Then at line 549 change `token_budget=config.token_budget` to `token_budget=effective_token_budget`.

- [ ] **Step 3: Modify inject_message path in agent_manager.py**

In the `inject_message()` method around line 767-780, derive token_budget when constructing AgentConfig:

```python
            from agent_os.agent.pricing import get_cost_rates, budget_usd_to_token_budget
            cost_per_1k_input, cost_per_1k_output = get_cost_rates(
                model, project.get("provider", "custom"),
            )
            derived_token_budget = budget_usd_to_token_budget(
                project.get("budget_limit_usd"), cost_per_1k_input, cost_per_1k_output,
            )
            config = AgentConfig(
                ...
                token_budget=derived_token_budget,
                ...
            )
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/unit/ tests/platform/ -q --ignore=tests/platform/test_consumer3_wiring.py`
Expected: 629+ passed, 0 failures

- [ ] **Step 5: Commit**

```bash
git add agent_os/daemon_v2/agent_manager.py tests/unit/test_token_budget_from_usd.py
git commit -m "feat: derive token_budget from dollar budget via model pricing"
```

---

### Task 4: Run full test suite and verify

- [ ] **Step 1: Run complete test suite**

Run: `python -m pytest tests/unit/ tests/platform/ -q --ignore=tests/platform/test_consumer3_wiring.py`
Expected: 629+ passed, 0 failures

- [ ] **Step 2: Verify the math end-to-end**

Quick sanity check script:
```python
from agent_os.agent.pricing import get_cost_rates, budget_usd_to_token_budget
rates = get_cost_rates("claude-sonnet-4-5", "anthropic")
print(f"Sonnet rates: {rates}")
budget = budget_usd_to_token_budget(5.0, *rates)
print(f"$5 budget → {budget:,} token budget")
# Should be ~1M+ tokens, much more generous than old 500K
```
