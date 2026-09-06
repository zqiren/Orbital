# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test doubles for spec 082 credential cards.

``_build_agent_config_from_project`` resolves the whole LLM half of a config
from ONE card, so a settings-store double has to answer card lookups. A bare
``MagicMock`` is worse than useless here: every attribute is truthy, so a
missing method silently returns a mock that flows into ``AgentConfig`` and
surfaces three layers away as ``model=<MagicMock ...>``.

``FakeCardStore`` implements exactly the surface the daemon calls and nothing
else, so a method the production code starts using shows up as an
``AttributeError`` in the test that needs it.
"""

from agent_os.daemon_v2.settings_store import (
    CredentialCard,
    FallbackModelConfig,
    GlobalSettings,
    default_card_name,
)


class FakeCardStore:
    """A card-aware settings store double."""

    def __init__(self, cards=None, keys=None, default_card_id=None,
                 settings=None):
        self.cards = list(cards or [])
        self.keys = dict(keys or {})
        self.default_card_id = default_card_id or (
            self.cards[0].id if self.cards else None
        )
        self._settings = settings if settings is not None else GlobalSettings()
        # Recorded side effects, for tests that assert on them.
        self.touched: list[str] = []
        self.health: list[tuple] = []

    # -- construction helpers -------------------------------------------

    @classmethod
    def with_default(cls, *, provider="openai", model="gpt-4o", key="k",
                     base_url=None, sdk=None, region="global",
                     card_id="card_default", name=None, settings=None):
        """A store holding exactly one card, which is the default."""
        store = cls(settings=settings)
        store.add(card_id=card_id, provider=provider, model=model, key=key,
                  base_url=base_url, sdk=sdk, region=region, name=name,
                  make_default=True)
        return store

    def add(self, *, card_id, provider="openai", model="gpt-4o", key="",
            base_url=None, sdk=None, region="global", name=None,
            make_default=False, last_error=None) -> CredentialCard:
        card = CredentialCard(
            id=card_id, name=name or default_card_name(provider, model),
            provider=provider, region=region, base_url=base_url, sdk=sdk,
            model=model, created_at="2026-01-01T00:00:00+00:00",
            last_error=last_error,
        )
        self.cards.append(card)
        self.keys[card_id] = key
        if make_default or self.default_card_id is None:
            self.default_card_id = card_id
        return card

    def set_global_fallbacks(self, card_ids) -> None:
        self._settings.llm.fallback_models = [
            FallbackModelConfig(card_id=cid) for cid in card_ids
        ]

    # -- the surface the daemon calls -----------------------------------

    def get(self) -> GlobalSettings:
        return self._settings

    def resolve_card(self, card_id):
        if not card_id:
            return None
        for card in self.cards:
            if card.id == card_id:
                return card
        return None

    get_card = resolve_card

    def stored_default_card(self):
        return self.resolve_card(self.default_card_id)

    default_card = stored_default_card

    def key_for(self, card_id) -> str:
        return self.keys.get(card_id, "") if card_id else ""

    def effective_default_card_id(self):
        return self.default_card_id

    def set_default_card(self, card_id) -> None:
        if self.resolve_card(card_id) is None:
            raise KeyError(card_id)
        self.default_card_id = card_id

    def touch_card_used(self, card_id) -> None:
        self.touched.append(card_id)

    def record_card_health(self, card_id, *, verified=False, error=None) -> None:
        self.health.append((card_id, verified, error))

    # -- writes (routes that mint or refresh a card) ---------------------

    def create_card(self, *, provider, model, name="", region="global",
                    base_url=None, sdk=None, api_key="", make_default=False):
        card_id = f"card_{len(self.cards):08x}"
        return self.add(card_id=card_id, provider=provider, model=model,
                        key=api_key, base_url=base_url, sdk=sdk, region=region,
                        name=name or None, make_default=make_default)

    def update_card(self, card_id, **fields):
        card = self.resolve_card(card_id)
        if card is None:
            raise KeyError(card_id)
        key = fields.pop("api_key", None)
        if key is not None:
            self.keys[card_id] = key
        for field, value in fields.items():
            setattr(card, field, value)
        return card

    def masked_card(self, card, **_kw) -> dict:
        """The card as a route returns it — never the raw key."""
        key = self.keys.get(card.id, "")
        return {
            "id": card.id, "name": card.name, "provider": card.provider,
            "region": card.region, "base_url": card.base_url, "sdk": card.sdk,
            "model": card.model, "created_at": card.created_at,
            "verified_at": getattr(card, "verified_at", None),
            "last_used_at": getattr(card, "last_used_at", None),
            "last_error": card.last_error,
            "key_set": bool(key),
            "key_masked": (key[:4] + "..." + key[-4:]) if len(key) > 8 else ("****" if key else ""),
            "key_source": "keychain" if key else "none",
            "is_default": card.id == self.default_card_id,
            "read_only": card.id == "env",
        }
