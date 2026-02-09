"""Chroma vector store wrapper for rules and actions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from theow._core._logging import get_logger
from theow._core._models import Rule

logger = get_logger(__name__)


class ChromaStore:
    """Wrapper around ChromaDB for rule and action storage."""

    def __init__(
        self,
        path: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._path = path
        self._embedding_model = embedding_model

        path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )

        self._actions_collection = self._client.get_or_create_collection(
            name="theow-actions",
            metadata={"hnsw:space": "cosine"},
        )

        # Cache of known metadata keys from indexed rules
        self._metadata_keys: set[str] = set()

    def _get_collection(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_rule(self, rule: Rule) -> None:
        """Index a rule into its collection."""
        collection = self._get_collection(rule.collection)

        metadata = rule.get_metadata()
        self._metadata_keys.update(
            k
            for k, v in metadata.items()
            if isinstance(v, str) and k not in ("type", "content_hash")
        )

        collection.upsert(
            ids=[rule.name],
            documents=[rule.get_embedding_text()],
            metadatas=[metadata],
        )
        logger.debug("Indexed rule", rule=rule.name, collection=rule.collection)

    def index_action(self, name: str, docstring: str, signature: str) -> None:
        """Index an action for semantic search."""
        embedding_text = f"{name}: {docstring}\nSignature: {signature}"

        self._actions_collection.upsert(
            ids=[name],
            documents=[embedding_text],
            metadatas=[{"docstring": docstring, "signature": signature}],
        )
        logger.debug("Indexed action", action=name)

    def sync_rules(self, rules_dir: Path) -> None:
        """Sync rules from directory, re-embedding only changed files.

        Also removes stale entries from chroma that no longer have files.
        """
        if not rules_dir.exists():
            return

        # Track which rules we sync (by collection)
        synced: dict[str, set[str]] = {}

        for rule_file in rules_dir.glob("*.rule.yaml"):
            rule = self._sync_rule_file(rule_file)
            if rule:
                synced.setdefault(rule.collection, set()).add(rule.name)

        # Remove stale entries from each collection
        for collection, rule_names in synced.items():
            self._cleanup_stale(collection, rule_names)

    def _cleanup_stale(self, collection: str, valid_names: set[str]) -> None:
        """Remove chroma entries that don't have corresponding files."""
        coll = self._get_collection(collection)
        existing = coll.get(include=[])
        stale = [name for name in existing["ids"] if name not in valid_names]
        if stale:
            coll.delete(ids=stale)
            logger.debug("Removed stale rules", collection=collection, rules=stale)

    def _sync_rule_file(self, path: Path) -> Rule | None:
        """Sync a single rule file. Returns the rule for tracking."""
        content = path.read_text()
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        rule = Rule.from_yaml(path)
        collection = self._get_collection(rule.collection)

        existing = collection.get(ids=[rule.name], include=["metadatas"])
        metadatas = existing.get("metadatas")

        if existing["ids"] and metadatas:
            stored_hash = metadatas[0].get("content_hash", "")
            if stored_hash == file_hash:
                logger.debug("Rule unchanged", rule=rule.name)
                return rule

        self.index_rule(rule)
        return rule

    def get_rule(self, collection: str, name: str) -> Rule | None:
        """Get a rule by name from a collection."""
        coll = self._get_collection(collection)
        result = coll.get(ids=[name], include=["documents", "metadatas"])

        if not result["ids"]:
            return None

        # We don't store full rule YAML in Chroma, only embeddings
        # Rules must be loaded from files; this just confirms existence
        return None

    def query_rules(
        self,
        collection: str,
        query_text: str,
        metadata_filter: dict[str, Any] | None = None,
        n_results: int = 10,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Query rules by semantic similarity.

        Returns list of (rule_name, distance, metadata).
        """
        coll = self._get_collection(collection)

        where = metadata_filter if metadata_filter else None

        try:
            results = coll.query(
                query_texts=[query_text],
                where=where,
                n_results=n_results,
                include=["distances", "metadatas"],
            )
        except Exception as e:
            logger.warning("Chroma query failed", error=str(e))
            return []

        ids = results.get("ids")
        distances = results.get("distances")
        metadatas = results.get("metadatas")

        if not ids or not ids[0] or not distances or not metadatas:
            return []

        return [(name, dist, meta) for name, dist, meta in zip(ids[0], distances[0], metadatas[0])]

    def query_actions(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Query actions by semantic similarity."""
        try:
            results = self._actions_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas"],
            )
        except Exception as e:
            logger.warning("Action query failed", error=str(e))
            return []

        ids = results.get("ids")
        metadatas = results.get("metadatas")

        if not ids or not ids[0] or not metadatas:
            return []

        return [{"name": name, **meta} for name, meta in zip(ids[0], metadatas[0])]

    def list_rules(self, collection: str) -> list[str]:
        """List all rule names in a collection."""
        coll = self._get_collection(collection)
        result = coll.get(include=[])
        return result["ids"]

    def list_actions(self) -> list[str]:
        """List all action names."""
        result = self._actions_collection.get(include=[])
        return result["ids"]

    def get_metadata_keys(self) -> set[str]:
        """Get known metadata keys for filtering."""
        return self._metadata_keys

    def update_rule_stats(
        self,
        collection: str,
        rule_name: str,
        success: bool,
        cost: float = 0.0,
    ) -> None:
        """Update success/fail counts and cost for a rule."""
        coll = self._get_collection(collection)

        existing = coll.get(ids=[rule_name], include=["metadatas", "documents"])
        metadatas = existing.get("metadatas")
        if not existing["ids"] or not metadatas:
            return

        old_meta = metadatas[0]
        new_meta: dict[str, Any] = dict(old_meta)

        if success:
            new_meta["success_count"] = int(new_meta.get("success_count", 0)) + 1
        else:
            new_meta["fail_count"] = int(new_meta.get("fail_count", 0)) + 1

        new_meta["cost"] = float(new_meta.get("cost", 0.0)) + cost

        coll.update(
            ids=[rule_name],
            metadatas=[new_meta],
        )

    def get_all_rules_with_stats(self) -> list[dict[str, Any]]:
        """Get all rules with their stats for meow()."""
        all_stats = []

        for coll_name in self._client.list_collections():
            if coll_name.name == "theow-actions":
                continue

            coll = self._client.get_collection(coll_name.name)
            result = coll.get(include=["metadatas"])
            metadatas = result.get("metadatas")
            if not metadatas:
                continue

            for name, meta in zip(result["ids"], metadatas):
                all_stats.append(
                    {
                        "name": name,
                        "collection": coll_name.name,
                        "success_count": meta.get("success_count", 0),
                        "fail_count": meta.get("fail_count", 0),
                        "explored": meta.get("explored", False),
                        "cost": meta.get("cost", 0.0),
                    }
                )

        return all_stats
