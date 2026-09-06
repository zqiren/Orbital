# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global settings store. Persists to {data_dir}/settings.json.

Spec 082 — credential cards. A *card* is one complete working LLM setup
(provider + endpoint choice + key + model), created with the form users
already know and verified on save. Cards live here as
``credential_cards: [...]`` plus ``default_card_id``; the raw keys live in
the OS keychain, one item per card (``CardKeyStore``). Everything that used
to hold credentials — the single global slot, the four-field per-project
override, every fallback entry — now holds a card id instead, so there is
nothing left to pair wrongly.
"""

import json
import logging
import os
from contextlib import contextmanager
import time
import sys
import shutil
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from agent_os.daemon_v2.credential_store import ENV_CARD_ID, CardKeyStore

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Distinguishes "leave default_card_id alone" from "set it to None".
_KEEP = object()


def _new_card_id() -> str:
    return "card_" + uuid4().hex[:8]


def mask_key(key: str | None) -> str:
    """First 4 + '...' + last 4, '****' for short keys, '' for none.

    The one masking rule for every surface (cards, the derived ``llm``
    block, TokenDance's legacy response fields) so a key can never leak
    through a surface that forgot to mask.
    """
    if not key:
        return ""
    return key[:4] + "..." + key[-4:] if len(key) > 8 else "****"


class FallbackModelConfig(BaseModel):
    """One entry of the GLOBAL fallback chain — a reference to a card.

    Spec 082 §3.3: fallback entries no longer carry their own
    provider/model/key/endpoint. Legacy five-field entries still LOAD (extra
    keys ignored) and are converted to cards by the one-shot migration.
    """
    model_config = ConfigDict(extra="ignore")

    card_id: str = ""


class CredentialCard(BaseModel):
    """A saved, verified working credential setup (spec 082 §3.1).

    ``model`` is part of the identity: a different model is a different card
    (D1). A MIGRATED card may carry ``model == ""`` and is flagged "needs a
    model" via ``last_error`` until the user edits one in — the key is
    preserved either way (D7).
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    name: str = ""
    provider: str = "custom"
    # Endpoint choice for a registry provider; ignored for "custom".
    region: str = "global"
    # Set for provider == "custom"; also preserved for a MIGRATED endpoint
    # that matches neither of the registry's two URLs for its provider.
    base_url: str | None = None
    sdk: str | None = None
    model: str = ""
    created_at: str = ""
    verified_at: str | None = None
    last_used_at: str | None = None
    # {"status": int|null, "code": str, "message": str, "at": iso} — written
    # by the save test, a manual Test, and the loop on 401/402/403. Displayed,
    # never acted on (spec 072 D2).
    last_error: dict | None = None


class GlobalLLMSettings(BaseModel):
    """Pre-cards global LLM block.

    Retained ONLY so a settings.json written before spec 082 still loads and
    can be migrated. Every field is dropped from disk on the first save after
    migration; ``get_masked`` serves a DERIVED ``llm`` block built from the
    default card instead.
    """
    model_config = ConfigDict(extra="ignore")

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    sdk: str = "openai"
    provider: str = "custom"
    fallback_models: list[FallbackModelConfig] = []


# Default FX pair rates. User-editable, static — NEVER fetched live. Keyed
# "{TARGET}_per_{BASE}". Only USD↔CNY is meaningful today.
DEFAULT_FX_RATES: dict[str, float] = {"CNY_per_USD": 7.2}


class GlobalSettings(BaseModel):
    llm: GlobalLLMSettings = GlobalLLMSettings()
    # Spec 082. ``None`` means "never migrated" and is the one-shot marker the
    # migration reads; after it runs the field is always a list (possibly
    # empty), so deleting every card can never re-trigger it.
    credential_cards: list[CredentialCard] | None = None
    default_card_id: str | None = None
    scratch_workspace: str | None = None
    user_preferences_path: str | None = None
    # Spec 073 — user-level memory: agent-written facts about the user,
    # injected into every project's prompt. A SEPARATE file from
    # user_preferences.md because the prefs save is a full overwrite (D2).
    # None resolves to ~/orbital/user_memory.md wherever the path is read.
    user_memory_path: str | None = None
    user_memory_enabled: bool = True
    # Budget Piece 1, Task 4 — static FX rates for cross-currency cost totals.
    # User-editable; no live lookups anywhere. Codes/numbers only.
    fx_rates: dict[str, float] = DEFAULT_FX_RATES.copy()
    # Spec 011 — BYO Google OAuth client for connectors. Until first-party app
    # registration lands, users supply their own client id/secret here; the
    # ConnectorManager's oauth_client_provider reads these at connect time.
    connector_google_client_id: str | None = None
    connector_google_client_secret: str | None = None
    # Spec 046 — aggregate-only telemetry. Default-on with first-run disclosure
    # + Global Settings toggle; the tolerant loader means existing installs
    # pick up the default. Only daily aggregates ever transmit (see
    # agent_os/telemetry/).
    telemetry_enabled: bool = True


# ---------------------------------------------------------------------------
# Card endpoint resolution
# ---------------------------------------------------------------------------

_REGISTRY = None


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        from agent_os.config.provider_registry import ProviderRegistry

        _REGISTRY = ProviderRegistry()
    return _REGISTRY


def provider_display_name(provider: str, registry=None) -> str:
    """The registry's display name for ``provider``, else the key itself."""
    data = (registry or _registry()).get_provider_data(provider) or {}
    return data.get("display_name") or provider


def default_card_name(provider: str, model: str, registry=None) -> str:
    """"<Provider display name> · <model>" — the default card name."""
    display = provider_display_name(provider, registry)
    return f"{display} · {model}" if model else display


def resolve_card_endpoint(card: CredentialCard, registry=None) -> tuple[str | None, str]:
    """Return ``(base_url, sdk)`` for ``card`` — the endpoint half of §3.4.

    A card's OWN ``base_url`` wins: that is the Custom/self-hosted escape
    hatch, and also how the migration preserves an endpoint that matches
    neither of the registry's two URLs for its provider. Otherwise the
    registry answers by region.
    """
    reg = registry or _registry()
    data = reg.get_provider_data(card.provider) or {}
    if card.base_url:
        base = card.base_url
    elif card.region == "china" and data.get("china_base_url"):
        base = data["china_base_url"]
    else:
        base = data.get("base_url")
    sdk = card.sdk or data.get("sdk") or "openai"
    return base, sdk


def region_for_base_url(provider: str, base_url: str | None,
                        registry=None) -> tuple[str, str | None]:
    """Classify a legacy ``base_url`` into ``(region, preserved_base_url)``.

    A URL equal to the registry's global or China endpoint becomes a region
    choice and is dropped from the card (so a later registry update moves the
    card with it). Anything else — a gateway, a proxy, a self-hosted box — is
    preserved verbatim on the card, because guessing it away would silently
    repoint a working setup.
    """
    reg = registry or _registry()
    data = reg.get_provider_data(provider) or {}
    if provider == "custom":
        return "global", base_url
    if not base_url:
        return "global", None
    trimmed = base_url.rstrip("/")
    china = (data.get("china_base_url") or "").rstrip("/")
    glob = (data.get("base_url") or "").rstrip("/")
    if china and trimmed == china:
        return "china", None
    if glob and trimmed == glob:
        return "global", None
    return "global", base_url


@contextmanager
def _migration_lock(data_dir: str, timeout: float = 60.0):
    """Hold an exclusive OS lock for the whole credential-card migration.

    Two daemons can start in the same second — the PID file lets a racer
    through whenever the recorded PID is dead, and on 2026-09-04 two instances
    started 377 ms apart on a user's machine. Both ran this one-shot
    migration. Each minted its OWN card ids, wrote Keychain items under them,
    and saved its own settings; the last writer won, so the surviving
    settings.json named one instance's ids while the Keychain held the
    other's. Every lookup missed, and because each instance's own writes
    succeeded, nothing logged an error. Meanwhile the plaintext project keys
    had already been stripped from projects.json — the only remaining copy.

    Blocking, not try-lock: the second instance must WAIT and then re-read, so
    it observes the first one's cards and returns without migrating. A
    timeout bounds the wait, because a dead holder must not wedge startup.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, ".migration.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "card migration: lock still held after %.0fs; proceeding "
                        "without it", timeout,
                    )
                    break
                time.sleep(0.1)
        yield acquired
    finally:
        if acquired:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


class SettingsStore:
    """Read/write global settings from a JSON file, and own the card list."""

    def __init__(self, data_dir: str, credential_store=None,
                 project_store=None, card_key_store=None, migrate: bool = True):
        self._path = os.path.join(data_dir, "settings.json")
        self._data_dir = data_dir
        self._credential_store = credential_store
        self._project_store = project_store
        self._cards = card_key_store if card_key_store is not None else CardKeyStore()
        # The pre-cards single-slot API (PUT /settings/api-key and friends)
        # is a shim over the DEFAULT card; wiring it here means no call site
        # has to know about the back-reference.
        if credential_store is not None and hasattr(credential_store, "bind_settings_store"):
            credential_store.bind_settings_store(self)
        if migrate:
            try:
                # Under an exclusive lock, and the "already migrated?" check
                # re-runs INSIDE it: a racing daemon that waited here must see
                # the winner's cards and do nothing, or both mint their own ids
                # and the surviving settings name cards whose keys were written
                # by the other process (the 2026-09-04 data loss).
                with _migration_lock(self._data_dir):
                    self.migrate_to_cards()
            except Exception:
                logger.exception(
                    "credential-card migration failed; leaving settings untouched"
                )

    # -- plain settings -------------------------------------------------

    def get(self) -> GlobalSettings:
        if not os.path.exists(self._path):
            return GlobalSettings()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return GlobalSettings(**data)
        except (json.JSONDecodeError, Exception):
            return GlobalSettings()

    def update(self, settings: GlobalSettings) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        data = settings.model_dump()
        # Legacy LLM fields are dropped on save (§3.7 step 5). The block itself
        # stays so a rollback still parses the file.
        data["llm"] = {"fallback_models": data.get("llm", {}).get("fallback_models", [])}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _read_raw(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_raw(self, data: dict) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # -- cards ----------------------------------------------------------

    def stored_cards(self) -> list[CredentialCard]:
        """The persisted cards, in stored order (no synthetic env card)."""
        return list(self.get().credential_cards or [])

    def _save_cards(self, cards: list[CredentialCard],
                    default_card_id=_KEEP) -> None:
        raw = self._read_raw()
        raw["credential_cards"] = [c.model_dump() for c in cards]
        if default_card_id is not _KEEP:
            raw["default_card_id"] = default_card_id
        self._write_raw(raw)

    def env_card(self) -> CredentialCard | None:
        """The synthetic read-only ``AGENT_OS_API_KEY`` card, or None (D11).

        It MIRRORS the stored default card's provider/model/endpoint, so the
        environment variable keeps meaning exactly what it always meant —
        "use this key for the global default" — rather than inventing a
        provider of its own.
        """
        if not self._cards.get(ENV_CARD_ID):
            return None
        src = self.stored_default_card()
        if src is not None:
            return CredentialCard(
                id=ENV_CARD_ID, name=src.name, provider=src.provider,
                region=src.region, base_url=src.base_url, sdk=src.sdk,
                model=src.model, created_at=src.created_at,
            )
        llm = self.get().llm
        provider = llm.provider or "custom"
        model = llm.model or ""
        region, base = region_for_base_url(provider, llm.base_url)
        return CredentialCard(
            id=ENV_CARD_ID,
            name=default_card_name(provider, model) if model else "Environment key",
            provider=provider, region=region, base_url=base,
            sdk=llm.sdk if provider == "custom" else None, model=model,
        )

    @staticmethod
    def _order_key(card: CredentialCard):
        # last_used_at desc (nulls last), then created_at asc.
        used = card.last_used_at or ""
        return (0 if used else 1, _Desc(used), card.created_at or "")

    def list_cards(self) -> list[CredentialCard]:
        """Every card the UI shows: the env card first, then recency order."""
        cards = sorted(self.stored_cards(), key=self._order_key)
        env = self.env_card()
        return ([env] if env is not None else []) + cards

    def get_card(self, card_id: str | None) -> CredentialCard | None:
        """Look up one card by id (``env`` included). None when unknown."""
        if not card_id:
            return None
        if card_id == ENV_CARD_ID:
            return self.env_card()
        for card in self.stored_cards():
            if card.id == card_id:
                return card
        return None

    # ``resolve_card`` is the name §3.4 uses; keep both so the config builder
    # reads like the spec.
    resolve_card = get_card

    def stored_default_card(self) -> CredentialCard | None:
        """The PERSISTED default card, ignoring the env override."""
        settings = self.get()
        cards = settings.credential_cards or []
        wanted = settings.default_card_id
        for card in cards:
            if card.id == wanted:
                return card
        return cards[0] if cards else None

    def default_card(self) -> CredentialCard | None:
        """The EFFECTIVE default: the env card while it exists, else stored."""
        return self.env_card() or self.stored_default_card()

    def effective_default_card_id(self) -> str | None:
        card = self.default_card()
        return card.id if card is not None else None

    def key_for(self, card_id: str | None) -> str:
        """The raw key for ``card_id``, or "" — the ONE key lookup."""
        return self._cards.get(card_id) or "" if card_id else ""

    def key_source(self, card_id: str | None) -> str:
        return self._cards.source(card_id) if card_id else "none"

    def masked_card(self, card: CredentialCard,
                    default_id: str | None = None) -> dict:
        """The wire shape of a card. The raw key is NEVER in it."""
        if default_id is None:
            default_id = self.effective_default_card_id()
        key = self.key_for(card.id)
        data = card.model_dump()
        data.update({
            "key_set": bool(key),
            "key_masked": mask_key(key),
            "key_source": self.key_source(card.id),
            "is_default": card.id == default_id,
            "read_only": card.id == ENV_CARD_ID,
        })
        return data

    def masked_cards(self) -> list[dict]:
        default_id = self.effective_default_card_id()
        return [self.masked_card(c, default_id) for c in self.list_cards()]

    # -- card mutation --------------------------------------------------

    def create_card(self, *, provider: str, model: str, name: str = "",
                    region: str = "global", base_url: str | None = None,
                    sdk: str | None = None, api_key: str | None = None,
                    make_default: bool | None = None) -> CredentialCard:
        """Persist a new card (and its key). The FIRST card becomes default."""
        settings = self.get()
        cards = list(settings.credential_cards or [])
        card = CredentialCard(
            id=_new_card_id(),
            name=name or default_card_name(provider, model),
            provider=provider, region=region or "global",
            base_url=base_url or None,
            sdk=sdk or None,
            model=model or "",
            created_at=_now_iso(),
        )
        if api_key is not None:
            self._cards.set(card.id, api_key)
        cards.append(card)
        should_default = make_default if make_default is not None else (
            settings.default_card_id is None or len(cards) == 1
        )
        self._save_cards(
            cards,
            card.id if should_default else settings.default_card_id,
        )
        return card

    def update_card(self, card_id: str, **fields) -> CredentialCard:
        """Patch a stored card. ``api_key`` (when present) replaces the key.

        The id never changes, so no project, fallback entry or session is
        touched by a re-key (D3).
        """
        if card_id == ENV_CARD_ID:
            raise ValueError("the environment card is read-only")
        cards = self.stored_cards()
        api_key = fields.pop("api_key", None)
        found = None
        for idx, card in enumerate(cards):
            if card.id != card_id:
                continue
            data = card.model_dump()
            for key, value in fields.items():
                if key in data:
                    data[key] = value
            found = CredentialCard(**data)
            cards[idx] = found
            break
        if found is None:
            raise KeyError(card_id)
        if api_key is not None:
            self._cards.set(card_id, api_key)
        self._save_cards(cards)
        return found

    def set_card_key(self, card_id: str, api_key: str) -> dict:
        if card_id == ENV_CARD_ID:
            return {"source": "environment"}
        return self._cards.set(card_id, api_key)

    def delete_card(self, card_id: str) -> None:
        """Remove a card and its key. Referrers are re-pointed by the route."""
        if card_id == ENV_CARD_ID:
            raise ValueError("the environment card is read-only")
        cards = [c for c in self.stored_cards() if c.id != card_id]
        settings = self.get()
        default_id = settings.default_card_id
        if default_id == card_id:
            default_id = cards[0].id if cards else None
        self._cards.delete(card_id)
        self._save_cards(cards, default_id)

    def set_default_card(self, card_id: str) -> None:
        raw = self._read_raw()
        raw["default_card_id"] = card_id
        self._write_raw(raw)

    def ensure_default_card(self) -> CredentialCard:
        """The stored default card, creating an empty one if there is none.

        The ``PUT /settings/api-key`` shim needs somewhere to put a key before
        the wizard's later ``PUT /settings`` supplies provider and model.
        """
        card = self.stored_default_card()
        if card is not None:
            return card
        llm = self.get().llm
        provider = llm.provider or "custom"
        region, base = region_for_base_url(provider, llm.base_url)
        return self.create_card(
            provider=provider, model=llm.model or "", region=region,
            base_url=base, sdk=llm.sdk if provider == "custom" else None,
            make_default=True,
        )

    def record_card_health(self, card_id: str, *, verified: bool = False,
                           error: dict | None = None) -> None:
        """Stamp ``verified_at`` or ``last_error`` on a card.

        A success clears the error; a failure leaves ``verified_at`` alone
        (it records when the card last WORKED, which stays true).
        """
        if not card_id or card_id == ENV_CARD_ID:
            return
        cards = self.stored_cards()
        changed = False
        for idx, card in enumerate(cards):
            if card.id != card_id:
                continue
            data = card.model_dump()
            if verified:
                data["verified_at"] = _now_iso()
                data["last_error"] = None
            else:
                data["last_error"] = error
            cards[idx] = CredentialCard(**data)
            changed = True
            break
        if changed:
            self._save_cards(cards)

    def touch_card_used(self, card_id: str | None) -> None:
        """Write ``last_used_at`` — once per run, at session start (§3.5)."""
        if not card_id or card_id == ENV_CARD_ID:
            return
        cards = self.stored_cards()
        for idx, card in enumerate(cards):
            if card.id != card_id:
                continue
            data = card.model_dump()
            data["last_used_at"] = _now_iso()
            cards[idx] = CredentialCard(**data)
            self._save_cards(cards)
            return

    # -- the derived masked view ----------------------------------------

    def _derived_llm(self) -> dict:
        """The pre-cards ``llm`` block, DERIVED from the default card.

        Kept for one release so ``App.tsx``, ``SubAgentSettings.tsx`` and any
        older SPA against a newer daemon keep reading provider/model/key
        status where they always did (§3.8, risk "two SPAs, one daemon").
        """
        card = self.default_card()
        settings = self.get()
        if card is None:
            return {
                "provider": "custom", "model": "", "base_url": None,
                "sdk": "openai", "api_key_set": False, "api_key_masked": "",
                "api_key_source": "none", "fallback_models": [],
            }
        base_url, sdk = resolve_card_endpoint(card)
        key = self.key_for(card.id)
        fallbacks = []
        for entry in (settings.llm.fallback_models or []):
            ref = self.get_card(entry.card_id)
            fallbacks.append({
                "card_id": entry.card_id,
                "provider": ref.provider if ref else None,
                "model": ref.model if ref else None,
            })
        return {
            "provider": card.provider,
            "model": card.model,
            "base_url": base_url,
            "sdk": sdk,
            "api_key_set": bool(key),
            "api_key_masked": mask_key(key),
            "api_key_source": self.key_source(card.id),
            "fallback_models": fallbacks,
        }

    def get_masked(self) -> dict:
        """Settings for the frontend: cards + the derived ``llm`` block."""
        settings = self.get()
        data = settings.model_dump()
        data["credential_cards"] = self.masked_cards()
        data["default_card_id"] = self.effective_default_card_id()
        data["llm"] = self._derived_llm()

        # Read user preferences file content
        prefs_path = data.get("user_preferences_path")
        if prefs_path and os.path.exists(prefs_path):
            try:
                with open(prefs_path, "r", encoding="utf-8") as f:
                    data["user_preferences_content"] = f.read()
            except OSError:
                data["user_preferences_content"] = ""
        else:
            data["user_preferences_content"] = ""

        # Read user memory file content (spec 073). Unlike prefs, the file is
        # agent-written and normally exists at the DEFAULT path before the
        # user ever saves anything from Settings, so resolve the default here
        # rather than requiring user_memory_path to have been set.
        memory_path = data.get("user_memory_path") or os.path.join(
            os.path.expanduser("~"), "orbital", "user_memory.md")
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r", encoding="utf-8") as f:
                    data["user_memory_content"] = f.read()
            except OSError:
                data["user_memory_content"] = ""
        else:
            data["user_memory_content"] = ""

        return data

    # -- migration (§3.7) -----------------------------------------------

    def migrate_to_cards(self) -> dict | None:
        """One-shot, idempotent, lossless migration to credential cards.

        Runs at construction — before any route serves — and is marked done by
        writing ``credential_cards`` (a list, possibly empty). Deleting every
        card afterwards can therefore never re-trigger it.

        Returns a summary dict when it ran, else None.
        """
        raw = self._read_raw()
        if raw.get("credential_cards") is not None:
            return None

        projects = self._project_store.list_projects() if self._project_store else []
        llm = raw.get("llm") or {}
        legacy_key = self._cards.get_legacy()
        had_anything = bool(
            raw or legacy_key
            or any(p.get("api_key") or p.get("model") for p in projects)
        )
        self._backup_before_migration(bool(had_anything))

        registry = _registry()
        cards: list[CredentialCard] = []
        # (provider, model, region, base_url, key) -> card id, so a shared key
        # never mints two identical cards.
        by_shape: dict[tuple, str] = {}
        key_writes: dict[str, str] = {}

        def mint(provider: str, model: str, base_url: str | None,
                 sdk: str | None, key: str, name: str,
                 error: dict | None = None) -> CredentialCard:
            region, kept_base = region_for_base_url(provider, base_url, registry)
            shape = (provider, model, region, kept_base, key)
            existing = by_shape.get(shape)
            if existing is not None:
                for c in cards:
                    if c.id == existing:
                        return c
            card = CredentialCard(
                id=_new_card_id(), name=name, provider=provider, region=region,
                base_url=kept_base or None,
                sdk=(sdk or None) if provider == "custom" else None,
                model=model, created_at=_now_iso(), last_error=error,
            )
            cards.append(card)
            by_shape[shape] = card.id
            key_writes[card.id] = key
            return card

        # 1. Global slot -> card G.
        g_provider = llm.get("provider") or "custom"
        g_model = llm.get("model") or ""
        g_base = llm.get("base_url")
        g_sdk = llm.get("sdk") or "openai"
        g_key = legacy_key or llm.get("api_key") or ""
        global_card = None
        if g_key or g_model:
            global_card = mint(
                g_provider, g_model, g_base, g_sdk, g_key,
                default_card_name(g_provider, g_model, registry),
                error=None if g_key else _card_error(
                    "missing_api_key", "No API key — add one to use this card.",
                ),
            )

        # 2/3. Projects.
        project_updates: dict[str, dict] = {}
        # pid -> the card that now owns THIS project's own plaintext key.
        project_key_cards: dict[str, str] = {}
        for project in projects:
            pid = project.get("project_id")
            if not pid:
                continue
            p_key = project.get("api_key") or ""
            p_model = project.get("model") or ""
            p_provider = project.get("provider") or "custom"
            p_base = project.get("base_url")
            p_sdk = project.get("sdk") or "openai"
            p_name = project.get("name") or pid
            update: dict = {"card_id": None, "migration_note": None}

            if p_key:
                # 2. Its own key: always becomes a card (D7, lossless).
                base_name = default_card_name(p_provider, p_model, registry)
                card = mint(
                    p_provider, p_model, p_base, p_sdk, p_key,
                    f"{base_name} ({p_name})",
                    error=None if p_model else _card_error(
                        "missing_model",
                        "This card needs a model before it can run.",
                    ),
                )
                # Remember whose plaintext key this card now owns, so the
                # row is only stripped if that key provably landed.
                project_key_cards[pid] = card.id
                if p_model:
                    update["card_id"] = card.id
                else:
                    # Key + endpoint but no model (the launch-vid shape): the
                    # card keeps the key and is flagged, while the project runs
                    # on the default card immediately.
                    update["migration_note"] = f"card_incomplete:{card.name}"
            elif p_model:
                # 3. A model but no key of its own.
                if global_card is not None and p_provider == global_card.provider:
                    if p_model == global_card.model:
                        pass  # identical to the default: follow it
                    else:
                        card = mint(
                            p_provider, p_model, p_base or g_base, p_sdk, g_key,
                            default_card_name(p_provider, p_model, registry),
                            error=None if g_key else _card_error(
                                "missing_api_key",
                                "No API key — add one to use this card.",
                            ),
                        )
                        update["card_id"] = card.id
                else:
                    update["migration_note"] = f"needs_card:{p_provider}/{p_model}"

            # 4b. Per-project fallback entries.
            update["llm_fallback_models"] = self._migrate_fallbacks(
                project.get("llm_fallback_models"), mint, global_card, g_key,
                registry,
            )
            project_updates[pid] = update

        # 4a. Global fallback entries.
        global_fallbacks = self._migrate_fallbacks(
            llm.get("fallback_models"), mint, global_card, g_key, registry,
        )

        # Write the keys, then VERIFY each one reads back. A silent no-op
        # backend (CI's null keyring, a locked keychain) accepts a write and
        # returns nothing, so 'no exception' is not evidence the key is
        # there — only a matching read-back is.
        key_errors: dict[str, dict] = {}
        for card_id, key in key_writes.items():
            if not key:
                continue
            try:
                self._cards.set(card_id, key)
                if self._cards.get(card_id) != key:
                    raise RuntimeError("read-back did not match")
            except Exception as exc:
                logger.error(
                    "card migration: could not store the key for card %s: %s",
                    card_id, exc,
                )
                key_errors[card_id] = _card_error(
                    "keychain_error",
                    "The key could not be written to the keychain — re-enter it.",
                )
        if key_errors:
            cards = [
                c if c.id not in key_errors
                else CredentialCard(**{**c.model_dump(),
                                       "last_error": key_errors[c.id]})
                for c in cards
            ]

        # The legacy Keychain item goes only after a verified read-back.
        if legacy_key and global_card is not None:
            if self._cards.get(global_card.id) == legacy_key:
                self._cards.delete_legacy()
            else:
                logger.error(
                    "card migration: read-back of card %s did not match the "
                    "legacy key; keeping the legacy keychain item",
                    global_card.id,
                )

        # 5. Persist: cards in, legacy LLM fields out.
        #
        raw["credential_cards"] = [c.model_dump() for c in cards]
        raw["default_card_id"] = global_card.id if global_card is not None else (
            cards[0].id if cards else None
        )
        raw["llm"] = {"fallback_models": global_fallbacks}
        self._write_raw(raw)

        # Saving a project row STRIPS its legacy llm fields, including the
        # plaintext api_key (`project_store._LEGACY_LLM_FIELDS`). So a row may
        # only be saved once its key is provably readable from the keychain
        # under the new card id. A project whose write failed keeps its
        # plaintext key and its old shape: it runs on the default provider,
        # its card is flagged, and the key is still there to recover from.
        # Lossy-but-visible was the old behaviour; this is not lossy at all.
        for pid, update in project_updates.items():
            if self._project_store is None:
                continue
            blocked = project_key_cards.get(pid)
            if blocked and blocked in key_errors:
                logger.error(
                    "card migration: keeping project %s untouched — its key "
                    "could not be stored under card %s, and saving the row "
                    "would drop the only copy",
                    pid, blocked,
                )
                continue
            self._project_store.update_project(pid, update)

        summary = {
            "cards": len(cards),
            "default_card_id": raw["default_card_id"],
            "projects": len(project_updates),
        }
        if cards or project_updates:
            logger.info("credential-card migration: %s", summary)
        return summary

    @staticmethod
    def _migrate_fallbacks(entries, mint, global_card, global_key,
                           registry) -> list[dict]:
        """Turn legacy five-field fallback entries into ``{card_id}`` refs."""
        out: list[dict] = []
        for entry in (entries or []):
            if not isinstance(entry, dict):
                continue
            if entry.get("card_id"):
                out.append({"card_id": entry["card_id"]})
                continue
            model = entry.get("model") or ""
            provider = entry.get("provider") or "custom"
            if not model:
                continue
            key = entry.get("api_key") or global_key
            if (global_card is not None and provider == global_card.provider
                    and model == global_card.model and not entry.get("api_key")):
                out.append({"card_id": global_card.id})
                continue
            card = mint(
                provider, model, entry.get("base_url"), entry.get("sdk"), key,
                default_card_name(provider, model, registry),
            )
            out.append({"card_id": card.id})
        return out

    def _backup_before_migration(self, anything: bool) -> None:
        """Copy settings.json / projects.json aside before the one-shot.

        The migration deletes the legacy Keychain item and drops legacy fields;
        a byte copy of what it started from makes the whole thing recoverable
        by hand if a shape we never saw slips through.
        """
        if not anything:
            return
        for name in ("settings.json", "projects.json"):
            src = os.path.join(self._data_dir, name)
            dst = src + ".pre-cards"
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    # These hold plaintext per-project keys. copy2 preserves the
                    # source mode, which is 0644 — readable by any process
                    # running as the user. Narrow it, and never let this file
                    # reach telemetry, the relay, or a support bundle.
                    os.chmod(dst, 0o600)
                    logger.info("card migration: backed up %s -> %s", name, dst)
                except OSError:
                    logger.warning("could not back up %s before migration", src)


def _card_error(code: str, message: str, status: int | None = None) -> dict:
    return {"status": status, "code": code, "message": message, "at": _now_iso()}


class _Desc:
    """Sort helper: reverse ordering for a string key inside a tuple."""

    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value

    def __lt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value
