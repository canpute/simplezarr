import json

from simplezarr.stores import MemoryStore
from simplezarr.nodes import open_zarr, ZarrGroup, ZarrArray
from simplezarr.utils.consolidated import consolidate_metadata

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

    # The returned dict has the expected shape
    assert meta["kind"] == "auto"
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
    assert root_meta["consolidated_metadata"]["kind"] == "auto"

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
    # The kind fields are all "auto"
    assert root_meta["consolidated_metadata"]["kind"] == "auto"
    assert sub_meta["consolidated_metadata"]["kind"] == "auto"


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


def test_consolidate_metadata_empty():
    store = MemoryStore()
    ZarrGroup.create(store, "")
    group = open_zarr(store)
    meta = consolidate_metadata(group)
    assert meta == {"kind": "auto", "metadata": {}}


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
