import json

from simplezarr.stores import MemoryStore
from simplezarr.nodes import open_zarr, ZarrGroup, ZarrArray
from simplezarr.utils.consolidated import consolidate_metadata

import numpy as np
import pytest


def create_test_zarr(store=None):
    """Create a small Zarr file with a nested structure:

    root/                     (group, attributes: {"spam": "ham"})
    root/array0               (array)
    root/sub/                 (group)
    root/sub/array1           (array)
    root/sub/group2/          (group)
    root/sub/group2/array2    (array)
    """
    store = MemoryStore() if store is None else store
    ZarrGroup.create(store, "", attributes={"spam": "ham"})
    ZarrArray.create(store, "array0", (8, 6), "uint16")
    ZarrGroup.create(store, "sub")
    ZarrArray.create(store, "sub/array1", (8, 6), "uint16", chunk_shape=(4, 3))
    ZarrGroup.create(store, "sub/group2")
    ZarrArray.create(store, "sub/group2/array2", (8, 6), "uint16")
    return store


def test_consolidate_metadata():
    store = create_test_zarr()
    group = open_zarr(store)

    meta = consolidate_metadata(group)

    # The returned dict follows the v3 spec
    assert meta["kind"] == "inline"
    assert meta["must_understand"] is False
    assert list(meta["metadata"].keys()) == [
        "array0",
        "sub",
        "sub/array1",
        "sub/group2",
        "sub/group2/array2",
    ]

    # Each entry is the full metadata of that node (as in its zarr.json)
    for path, metadata in meta["metadata"].items():
        on_disk = json.loads(store.get(f"{path}/zarr.json").decode())
        assert metadata == on_disk, path

    # The root itself is not included, only its descendants


def test_consolidate_metadata_is_written():
    store = create_test_zarr()
    group = open_zarr(store)

    consolidate_metadata(group)

    # The zarr.json on disk contains the consolidated metadata ...
    root_meta = json.loads(store.get("zarr.json").decode())
    assert "consolidated_metadata" in root_meta
    assert list(root_meta["consolidated_metadata"]["metadata"].keys()) == [
        "array0",
        "sub",
        "sub/array1",
        "sub/group2",
        "sub/group2/array2",
    ]
    # ... and the root's other metadata is untouched
    assert root_meta["zarr_format"] == 3
    assert root_meta["node_type"] == "group"
    assert root_meta["attributes"] == {"spam": "ham"}
    assert root_meta["consolidated_metadata"]["kind"] == "inline"
    assert root_meta["consolidated_metadata"]["must_understand"] is False

    # The in-memory node is in sync with what is on disk
    assert group.metadata == root_meta


def test_consolidate_metadata_twice():
    # Consolidating again replaces the old entry (and does not duplicate it)
    store = create_test_zarr()
    group = open_zarr(store)
    consolidate_metadata(group)

    # Add a new node, then consolidate again
    ZarrArray.create(store, "sub/array3", (8, 6), "uint16")
    group = open_zarr(store)  # re-open, to pick up the new child
    consolidate_metadata(group)

    root_meta = json.loads(store.get("zarr.json").decode())
    meta = root_meta["consolidated_metadata"]["metadata"]
    assert list(meta.keys()) == [
        "array0",
        "sub",
        "sub/array1",
        "sub/array3",
        "sub/group2",
        "sub/group2/array2",
    ]
    assert "sub/array3" in meta


def test_consolidate_metadata_stale_sub_entry_dropped():
    # If a sub-group already has consolidated metadata (e.g. written when it
    # was the top group), it is dropped, because the new entry covers it
    store = create_test_zarr()
    sub = open_zarr(store, "sub")
    consolidate_metadata(sub)
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    assert "consolidated_metadata" in sub_meta

    group = open_zarr(store)
    consolidate_metadata(group)

    root_meta = json.loads(store.get("zarr.json").decode())
    meta = root_meta["consolidated_metadata"]["metadata"]
    assert "consolidated_metadata" not in meta["sub"]
    assert "consolidated_metadata" not in meta["sub/group2"]


def test_consolidate_metadata_sub_group():
    # Consolidating a sub-group produces paths relative to that group
    store = create_test_zarr()
    sub = open_zarr(store, "sub")
    meta = consolidate_metadata(sub)
    assert list(meta["metadata"].keys()) == [
        "array1",
        "group2",
        "group2/array2",
    ]
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    assert "consolidated_metadata" in sub_meta


def test_consolidate_metadata_recursive():
    store = create_test_zarr()
    group = open_zarr(store)

    top_meta = consolidate_metadata(group, recursive=True)

    # The return value is the entry of the top group (unchanged behavior)
    assert list(top_meta["metadata"].keys()) == [
        "array0",
        "sub",
        "sub/array1",
        "sub/group2",
        "sub/group2/array2",
    ]

    # Each descending group has its own entry, with paths relative to it
    root_meta = json.loads(store.get("zarr.json").decode())
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    group2_meta = json.loads(store.get("sub/group2/zarr.json").decode())

    assert list(root_meta["consolidated_metadata"]["metadata"].keys()) == [
        "array0",
        "sub",
        "sub/array1",
        "sub/group2",
        "sub/group2/array2",
    ]
    assert list(sub_meta["consolidated_metadata"]["metadata"].keys()) == [
        "array1",
        "group2",
        "group2/array2",
    ]
    assert list(group2_meta["consolidated_metadata"]["metadata"].keys()) == [
        "array2",
    ]

    # Each sub-entry matches the zarr.json of that node, modulo the
    # consolidated metadata that the recursive writing adds to sub-groups
    for rel_path, metadata in sub_meta["consolidated_metadata"]["metadata"].items():
        on_disk = json.loads(store.get(f"sub/{rel_path}/zarr.json").decode())
        on_disk.pop("consolidated_metadata", None)
        assert metadata == on_disk, rel_path

    # The groups' other metadata is preserved, also that of the sub-groups
    assert sub_meta["zarr_format"] == 3
    assert sub_meta["node_type"] == "group"

    # Arrays do not get consolidated metadata
    array0_meta = json.loads(store.get("array0/zarr.json").decode())
    assert "consolidated_metadata" not in array0_meta


def test_consolidate_metadata_recursive_twice():
    # A second (recursive) consolidation replaces all previous entries
    store = create_test_zarr()
    group = open_zarr(store)
    consolidate_metadata(group, recursive=True)

    ZarrArray.create(store, "sub/array3", (8, 6), "uint16")
    group = open_zarr(store)
    consolidate_metadata(group, recursive=True)

    root_meta = json.loads(store.get("zarr.json").decode())
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    assert "sub/array3" in root_meta["consolidated_metadata"]["metadata"]
    assert "array3" in sub_meta["consolidated_metadata"]["metadata"]
    assert list(sub_meta.keys()).count("consolidated_metadata") == 1
    # The kind fields are all "inline", per the spec
    assert root_meta["consolidated_metadata"]["kind"] == "inline"
    assert root_meta["consolidated_metadata"]["must_understand"] is False
    assert sub_meta["consolidated_metadata"]["kind"] == "inline"


def test_consolidate_metadata_default_not_recursive():
    # Without recursive=True, the sub-groups are not touched
    store = create_test_zarr()
    group = open_zarr(store)
    consolidate_metadata(group)

    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    group2_meta = json.loads(store.get("sub/group2/zarr.json").decode())
    assert "consolidated_metadata" not in sub_meta
    assert "consolidated_metadata" not in group2_meta

    # And the entry at the top correctly drops nothing (the sub-group
    # metadata shown there is the plain group metadata)
    root_meta = json.loads(store.get("zarr.json").decode())
    assert "consolidated_metadata" not in root_meta["consolidated_metadata"]["metadata"][
        "sub"
    ]


# %%%%% The following tests are focused on making sure that consolidation
# can never corrupt existing metadata.


def _store_snapshot(store):
    return {key: bytes(value) for key, value in store.dict.items()}


class FailOnSetStore(MemoryStore):
    """A MemoryStore that fails all writes to ``fail_on_key`` (when set)."""

    def __init__(self, fields=None):
        self.fail_on_key = None
        super().__init__(fields)

    def set(self, key, value):
        if key == self.fail_on_key:
            raise IOError(f"Cannot write {key!r} (simulated failure)")
        return super().set(key, value)


def _gnarly_value():
    # A metadata value with all the JSON edge cases we can think of
    return {
        "str": "ünïcödé with \" quotes and a 🌍",
        "int": 42,
        "float": 1.25e-300,
        "bool_true": True,
        "bool_false": False,
        "null": None,
        "list": [1, "two", None, [True, {"nested": ["deeply"]}]],
        "dict": {"b": 2, "a": 1, "": "empty key"},
        "float_int_zero": 0.0,
    }


def test_consolidate_preserves_gnarly_metadata():
    store = create_test_zarr()
    # Add all kinds of tricky values, at the top and in the sub-groups
    ZarrGroup.create(
        store, "", attributes={"spam": "ham", "gnarly": _gnarly_value()}
    )
    ZarrGroup.create(store, "sub", attributes={"gnarly": _gnarly_value()})
    ZarrArray.create(store, "array0", (8, 6), "uint16", attributes={"gnarly": _gnarly_value()})

    group = open_zarr(store)
    before = {key: store.get(key) for key in
              ["zarr.json", "sub/zarr.json", "sub/group2/zarr.json", "array0/zarr.json"]}

    consolidate_metadata(group, recursive=True)

    for key, before_text in before.items():
        meta_before = json.loads(before_text.decode())
        meta_after = json.loads(store.get(key).decode())
        meta_after.pop("consolidated_metadata", None)
        meta_before.pop("consolidated_metadata", None)
        assert meta_after == meta_before, key

    # Verify that the gnarly values survive the round trip through the
    # consolidated entries too
    root_meta = json.loads(store.get("zarr.json").decode())
    sub_entry = root_meta["consolidated_metadata"]["metadata"]["sub"]
    assert sub_entry["attributes"]["gnarly"] == _gnarly_value()
    assert (
        root_meta["attributes"]["gnarly"] == _gnarly_value()
    )  # the root keeps its own attributes


def test_consolidate_touched_files_only():
    # Non-recursive: only the top zarr.json is modified, nothing else
    store = create_test_zarr()
    # Add a chunk (binary data) to make sure that it is not touched
    open_zarr(store)["sub/array1"].set_chunk_now((0, 0), np.ones((4, 3), "uint16"))

    before = _store_snapshot(store)
    consolidate_metadata(open_zarr(store))
    after = _store_snapshot(store)

    changed = [key for key in before if before[key] != after[key]]
    assert changed == ["zarr.json"]
    assert not (set(after) - set(before))  # no new files

    # Recursive: only the zarr.jsons of the groups are modified
    before = _store_snapshot(store)
    consolidate_metadata(open_zarr(store), recursive=True)
    after = _store_snapshot(store)

    changed = [key for key in before if before[key] != after[key]]
    assert changed == ["sub/zarr.json", "sub/group2/zarr.json"]  # top already had it
    assert not (set(after) - set(before))

    # After consolidation, the file opens fine, with all metadata valid
    group = open_zarr(store)
    assert group["sub/group2/array2"].shape == (8, 6)


def test_consolidate_metadata_idempotent():
    # A second consolidation (non-recursive and recursive) writes the exact
    # same bytes as the first, no drift
    store = create_test_zarr()
    consolidate_metadata(open_zarr(store))
    consolidate_metadata(open_zarr(store), recursive=True)
    snap1 = _store_snapshot(store)
    consolidate_metadata(open_zarr(store))
    consolidate_metadata(open_zarr(store), recursive=True)
    snap2 = _store_snapshot(store)
    assert snap1 == snap2


def test_consolidate_metadata_isolated():
    # The returned metadata shares no state with the store or the nodes:
    # mutating it cannot corrupt anything
    store = create_test_zarr()
    group = open_zarr(store)
    returned = consolidate_metadata(group)

    # Mutate the returned structure
    returned["kind"] = "MUTATED"
    returned["metadata"]["sub"]["node_type"] = "MUTATED"
    returned["metadata"]["sub"]["injected"] = True
    returned["metadata"].clear()

    # The store is untouched by the mutations
    rooted = json.loads(store.get("zarr.json").decode())
    assert rooted["consolidated_metadata"]["kind"] == "inline"
    assert rooted["consolidated_metadata"]["metadata"]["sub"]["node_type"] == "group"
    sub_attr = rooted["consolidated_metadata"]["metadata"]["sub"].get("attributes", {})
    assert "injected" not in sub_attr

    # Mutating the node does not affect the store
    group.metadata["attributes"]["injected"] = True
    rooted = json.loads(store.get("zarr.json").decode())
    assert "injected" not in rooted["attributes"]


def test_consolidate_metadata_uses_fresh_metadata():
    # If a zarr.json was modified in the store after the group was opened,
    # the fresh content is consolidated, and never clobbered with the stale
    store = create_test_zarr()
    group = open_zarr(store)  # the tree of this object is now stale
    assert "fresh" not in group["sub"].attributes

    # Modify the sub-group's metadata directly in the store
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    sub_meta["attributes"] = {"fresh": True}
    store.set("sub/zarr.json", json.dumps(sub_meta, indent=4).encode())

    # Non-recursive: the new attributes appear in the consolidated entry,
    # and the sub-group's zarr.json is untouched
    returned = consolidate_metadata(group)
    assert returned["metadata"]["sub"]["attributes"] == {"fresh": True}
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    assert "consolidated_metadata" not in sub_meta
    assert sub_meta["attributes"] == {"fresh": True}

    # Recursive mode: the sub-group's zarr.json is re-written, but based on
    # the fresh content (the stale attributes are NOT written back!)
    consolidate_metadata(group, recursive=True)
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    assert sub_meta["attributes"] == {"fresh": True}
    assert "consolidated_metadata" in sub_meta
    assert sub_meta["consolidated_metadata"]["kind"] == "inline"


def test_consolidate_recursive_failure_leaves_top_untouched():
    # If a sub-group write fails during recursive consolidation, the top
    # zarr.json is not written at all (the top is written last), so the
    # store never carries a top-level promise that something else lacks.
    store = FailOnSetStore()
    create_test_zarr(store)

    group = open_zarr(store)
    consolidate_metadata(group)  # Non-recursive is fine
    store.fail_on_key = "sub/group2/zarr.json"  # enable the failure
    before = _store_snapshot(store)

    with pytest.raises(IOError, match="simulated failure"):
        consolidate_metadata(group, recursive=True)

    # The sub-group that failed to write is simply not consolidated ...
    group2_meta = json.loads(store.get("sub/group2/zarr.json").decode())
    assert "consolidated_metadata" not in group2_meta
    # ... and the top is fully unchanged (byte-identical)
    after = _store_snapshot(store)
    changed = [key for key in before if before[key] != after[key]]
    assert changed == ["sub/zarr.json"]  # the sub written before the failure
    assert json.loads(store.get("zarr.json").decode()) == json.loads(before["zarr.json"])

    # The damage is recoverable: after solving the write problem,
    # consolidation succeeds and everything is consistent again
    store.fail_on_key = None
    consolidate_metadata(open_zarr(store), recursive=True)
    for path in ["zarr.json", "sub/zarr.json", "sub/group2/zarr.json"]:
        meta = json.loads(store.get(path).decode())
        assert meta["consolidated_metadata"]["kind"] == "inline"
        assert meta["consolidated_metadata"]["must_understand"] is False
        assert isinstance(meta["consolidated_metadata"]["metadata"], dict)


def test_consolidate_attributes_named_consolidated():
    # User data that happens to be named "consolidated_metadata" (inside
    # attributes) is preserved; only the top-level key is replaced
    attributes = {"note": "user data", "consolidated_metadata": {"user": "data"}}
    ZarrGroup.create(store := MemoryStore(), "", attributes=attributes)
    ZarrGroup.create(store, "sub", attributes=attributes)

    consolidate_metadata(open_zarr(store), recursive=True)

    for path in ["", "sub"]:
        meta = json.loads(store.get(f"{path}/zarr.json".lstrip("/")).decode())
        assert meta["attributes"]["consolidated_metadata"] == {"user": "data"}
        assert meta["consolidated_metadata"]["kind"] == "inline"
        assert meta["consolidated_metadata"]["must_understand"] is False
    # The sub-group's entry in the top-level consolidation contains the
    # user's attributes too
    root_meta = json.loads(store.get("zarr.json").decode())
    assert (
        root_meta["consolidated_metadata"]["metadata"]["sub"]["attributes"][
            "consolidated_metadata"
        ]
        == {"user": "data"}
    )


def test_consolidate_unicode_names():
    store = MemoryStore()
    ZarrGroup.create(store, "", attributes={"label": "ünïcödé 🌍"})
    ZarrArray.create(store, "λ-array", (8, 6), "uint16")
    ZarrGroup.create(store, "λ-group", attributes={"label": "gr"})
    ZarrGroup.create(store, "λ-group/sub", attributes={"emoji": "🌍"})

    returned = consolidate_metadata(open_zarr(store))

    assert list(returned["metadata"].keys()) == ["λ-array", "λ-group", "λ-group/sub"]
    root_meta = json.loads(store.get("zarr.json").decode())
    assert root_meta["attributes"] == {"label": "ünïcödé 🌍"}
    assert root_meta["consolidated_metadata"]["metadata"]["λ-group"]["attributes"] == {
        "label": "gr"
    }
    assert root_meta["consolidated_metadata"]["metadata"]["λ-group/sub"][
        "attributes"
    ] == {"emoji": "🌍"}

    # The file still opens, and the consolidated entry matches a fresh read
    group = open_zarr(store)
    assert group["λ-group/sub"].attributes == {"emoji": "🌍"}
    assert (
        returned["metadata"]["λ-array"]
        == json.loads(store.get("λ-array/zarr.json").decode())
    )


# %%%%% Reading via consolidated metadata


class CountingStore(MemoryStore):
    """A MemoryStore that counts the number of store requests per operation."""

    def __init__(self):
        self.requests = {}
        super().__init__()

    def _tick(self, name):
        self.requests[name] = self.requests.get(name, 0) + 1

    def _get(self, key):
        self._tick("get")
        return super()._get(key)

    def _get_partial_values(self, key_ranges):
        self._tick("get_partial_values")
        return super()._get_partial_values(key_ranges)

    def _list(self):
        self._tick("list")
        return super()._list()

    def _list_dir(self, prefix):
        self._tick("list_dir")
        return super()._list_dir(prefix)

    @property
    def n_requests(self):
        return sum(self.requests.values())


def _compare_trees(node1, node2):
    """Assert that two (freshly) opened trees are the same (modulo the
    consolidated metadata that groups may have).
    """
    assert node1.__class__ is node2.__class__
    assert node1.path == node2.path
    meta1 = {k: v for k, v in node1.metadata.items() if k != "consolidated_metadata"}
    meta2 = {k: v for k, v in node2.metadata.items() if k != "consolidated_metadata"}
    assert meta1 == meta2, node1.path
    assert node1.attributes == node2.attributes
    if isinstance(node1, ZarrGroup):
        names1 = [child.name for child in node1.children]
        names2 = [child.name for child in node2.children]
        assert names1 == names2, node1.path
        for child1, child2 in zip(node1.children, node2.children, strict=True):
            _compare_trees(child1, child2)


def test_open_with_consolidated_metadata():
    # Classic opening: a request per node (plus listing)
    store = CountingStore()
    create_test_zarr(store)
    store.requests.clear()
    classic = open_zarr(store)
    assert store.n_requests > 6

    # With consolidated metadata: exactly one request (the root zarr.json)
    consolidate_metadata(classic)
    store.requests.clear()
    fresh = open_zarr(store)
    assert store.requests == {"get": 1}

    # The resulting tree is identical (modulo the consolidated entry)
    _compare_trees(classic, fresh)


def test_open_with_consolidated_metadata_recursive():
    # Recursive consolidation: also a sub-group can be opened in one request
    store = CountingStore()
    create_test_zarr(store)
    consolidate_metadata(open_zarr(store), recursive=True)

    store.requests.clear()
    sub = open_zarr(store, "sub")
    assert store.requests == {"get": 1}
    assert [child.name for child in sub.children] == ["array1", "group2"]
    assert sub["group2/array2"].shape == (8, 6)

    # A grandchild group likewise
    store.requests.clear()
    group2 = open_zarr(store, "sub/group2")
    assert store.requests == {"get": 1}
    assert [child.name for child in group2.children] == ["array2"]


def test_open_with_consolidated_metadata_data_reads():
    # Opening via consolidated metadata does not affect reading data
    store = CountingStore()
    create_test_zarr(store)
    open_zarr(store)["sub/array1"].set_chunk_now((0, 0), np.arange(12, dtype="uint16").reshape(4, 3))
    consolidate_metadata(open_zarr(store))

    fresh = open_zarr(store)
    store.requests.clear()
    chunk = fresh["sub/array1"].get_chunk_now((0, 0))
    np.testing.assert_array_equal(chunk, np.arange(12, dtype="uint16").reshape(4, 3))
    # The chunk is loaded with a plain get (which allows e.g. SFTP prefetch)
    assert store.requests == {"get": 1}


def test_consolidated_open_trusts_metadata():
    # A node that was added to the store after consolidation is not seen
    # when the (stale) consolidated metadata is used; it is trusted as-is.
    store = create_test_zarr()
    consolidate_metadata(open_zarr(store))
    ZarrArray.create(store, "new-array", (8, 6), "uint16")

    group = open_zarr(store)
    assert "new-array" not in [child.name for child in group.children]

    # After re-consolidation it is there
    consolidate_metadata(group)
    group = open_zarr(store)
    assert "new-array" in [child.name for child in group.children]
    assert group["new-array"].shape == (8, 6)


def test_consolidated_malformed():
    # Cases that must raise, rather than silently dropping nodes
    def set_root_consolidated(store, payload):
        meta = json.loads(store.get("zarr.json").decode())
        meta["consolidated_metadata"] = payload
        store.set("zarr.json", json.dumps(meta).encode())

    # Payload that is not a dict, and one without the metadata field
    store = create_test_zarr()
    set_root_consolidated(store, "not a dict")
    with pytest.raises(RuntimeError, match="malformed"):
        open_zarr(store)

    store = create_test_zarr()
    set_root_consolidated(store, {"kind": "inline"})
    with pytest.raises(RuntimeError, match="malformed"):
        open_zarr(store)

    # Entry with an unexpected node_type
    store = create_test_zarr()
    set_root_consolidated(
        store,
        {"kind": "inline", "metadata": {"junk": {"zarr_format": 3, "node_type": "junk"}}},
    )
    with pytest.raises(RuntimeError, match="Unexpected node type"):
        open_zarr(store)

    # Nodes listed below an array
    store = create_test_zarr()
    entries = consolidate_metadata(open_zarr(store))["metadata"]
    entries["array0/spam"] = {"zarr_format": 3, "node_type": "group"}
    set_root_consolidated(store, {"kind": "inline", "metadata": entries})
    with pytest.raises(RuntimeError, match="below"):
        open_zarr(store)

    # Nodes listed below a node that has no entry itself
    store = create_test_zarr()
    set_root_consolidated(
        store,
        {"kind": "inline", "metadata": {"ghost/spam": {"zarr_format": 3, "node_type": "array"}}},
    )
    with pytest.raises(RuntimeError, match="non-existing"):
        open_zarr(store)

    # A node without any metadata
    store = create_test_zarr()
    set_root_consolidated(store, {"kind": "inline", "metadata": {"": "nope"}})
    with pytest.raises(RuntimeError, match="Malformed key"):
        open_zarr(store)

    # The malformed entries above all raised an error, but a normal
    # consolidated file still opens fine
    store = create_test_zarr()
    consolidate_metadata(open_zarr(store))
    group = open_zarr(store)
    assert [child.name for child in group.children] == ["array0", "sub"]


def test_consolidated_matches_spec():
    # Check the written consolidated metadata against the v3 spec
    # (docs/v3/core/v3.0.rst, "consolidated_metadata" section)
    store = create_test_zarr()
    consolidate_metadata(open_zarr(store), recursive=True)

    for path in ["", "sub", "sub/group2"]:
        meta = json.loads(store.get(f"{path}/zarr.json".lstrip("/")).decode())
        consolidated = meta["consolidated_metadata"]

        # The "kind" field must be the string literal 'inline'
        assert consolidated["kind"] == "inline"

        # The "must_understand" field must be the boolean literal False,
        # so that readers that don't know about consolidated metadata
        # can ignore it rather than failing to open the file
        assert consolidated["must_understand"] is False

        # The metadata must be a flat map, with the paths of ALL children
        # (recursively), relative to the node at which it is consolidated
        for key, node_meta in consolidated["metadata"].items():
            assert isinstance(key, str), key
            assert node_meta["zarr_format"] == 3
            assert node_meta["node_type"] in ("group", "array")

    # Keys relative to the consolidating group (spec examples B and C)
    root_meta = json.loads(store.get("zarr.json").decode())
    sub_meta = json.loads(store.get("sub/zarr.json").decode())
    group2_meta = json.loads(store.get("sub/group2/zarr.json").decode())
    assert "sub/group2/array2" in root_meta["consolidated_metadata"]["metadata"]
    assert "group2/array2" in sub_meta["consolidated_metadata"]["metadata"]
    assert "array2" in group2_meta["consolidated_metadata"]["metadata"]
    # ... and no node list itself
    assert all(
        key not in root_meta["consolidated_metadata"]["metadata"]
        for key in ["", ".", "/", "zarr.json"]
    )


def test_consolidated_kind_validation():
    # Legacy files (written by earlier versions of this library, with
    # kind "auto") and files without a kind still open
    def set_root_consolidated(store, payload):
        meta = json.loads(store.get("zarr.json").decode())
        meta["consolidated_metadata"] = payload
        store.set("zarr.json", json.dumps(meta).encode())

    store = create_test_zarr()
    consolidate_metadata(open_zarr(store))
    entries = json.loads(store.get("zarr.json").decode())["consolidated_metadata"]
    entries.pop("kind")
    set_root_consolidated(store, entries)
    group = open_zarr(store)
    assert [child.name for child in group.children] == ["array0", "sub"]

    store = create_test_zarr()
    consolidate_metadata(open_zarr(store))
    entries = json.loads(store.get("zarr.json").decode())["consolidated_metadata"]
    entries["kind"] = "auto"
    set_root_consolidated(store, entries)
    group = open_zarr(store)
    assert [child.name for child in group.children] == ["array0", "sub"]

    # But an unsupported kind raises, rather than being misinterpreted
    store = create_test_zarr()
    consolidate_metadata(open_zarr(store))
    entries = json.loads(store.get("zarr.json").decode())["consolidated_metadata"]
    entries["kind"] = "remote"  # a possible future kind
    set_root_consolidated(store, entries)
    with pytest.raises(RuntimeError, match="kind"):
        open_zarr(store)


def test_consolidate_metadata_empty():
    store = MemoryStore()
    ZarrGroup.create(store, "")
    group = open_zarr(store)
    meta = consolidate_metadata(group)
    assert meta == {"kind": "inline", "must_understand": False, "metadata": {}}


def test_consolidate_metadata_checks():
    store = create_test_zarr()

    # Not a group
    array = open_zarr(store, "array0")
    assert isinstance(array, ZarrArray)
    with pytest.raises(TypeError):
        consolidate_metadata(array)

    # Not a writable store
    class ReadOnlyish:
        pass

    group = open_zarr(store)
    fake_store = ReadOnlyish()
    group._store = fake_store
    with pytest.raises(TypeError):
        consolidate_metadata(group)
    group._store = store
