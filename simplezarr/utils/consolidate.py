# Utilities to write consolidated metadata for a Zarr group, in a format
# compatible with the convention used by zarr-python / xarray.
#
# A Zarr file is a tree of nodes, where each node has its own zarr.json.
# Opening a file (especially over a slow connection) therefore costs one
# store request per node. Consolidated metadata solves this: the metadata of
# all descending nodes is written into the zarr.json of the top group, so
# that the whole tree can be known from a single request.
#
# The consolidated metadata is stored as an entry in the group's metadata,
# following the v3 spec (see the "consolidated_metadata" section of
# https://zarr-specs.readthedocs.io)::
#
#     "consolidated_metadata": {
#         "kind": "inline",
#         "must_understand": false,
#         "metadata": {
#             "array1": {...},
#             "sub/array2": {...},
#             ...
#         }
#     }
#
# The keys in ``metadata`` are the paths of the nodes relative to the node at
# which the metadata is consolidated (i.e. the top group, which itself is not
# included), and the values are the full metadata of each node (i.e. the
# content of its zarr.json). The ``must_understand`` field signals to readers
# that don't know about consolidated metadata that they may ignore it.
#
# This code is written with a strong focus on never corrupting existing
# metadata. It therefore guarantees:
#
# * The only modification made to the store is adding/replacing the
#   "consolidated_metadata" entry in the zarr.json of the top group (and,
#   in recursive mode, of each descending group). All other keys of those
#   zarr.jsons (i.e. the rest of the metadata) are preserved as-is. No other
#   files in the store are touched.
# * The metadata is collected with a fresh read of the store (not using the
#   possibly stale in-memory tree of the given group), so changes written
#   after the group was opened are picked up rather than clobbered.
# * The collected metadata is deep-copied, so the written / returned data
#   cannot affect the in-memory nodes, and vice versa.
# * When anything fails, nothing (recursive mode) or no top-level promise
#   (non-recursive mode) is written: in recursive mode the top group is
#   written last, so a failure during the sub-group writes leaves the top
#   zarr.json untouched.

from __future__ import annotations

import copy
import json

from ..nodes import ZarrGroup, ZarrNode, join
from ..stores import WritableStore

__all__ = ["consolidate_metadata"]


def _inline_metadata(entries: dict) -> dict:
    # Build the consolidated metadata object, as defined by the v3 spec.
    # The "kind" field is reserved for future use ("inline" is the only
    # supported value), and "must_understand: false" allows readers that
    # don't know about consolidated metadata to still open the file.
    return {"kind": "inline", "must_understand": False, "metadata": entries}


def consolidate_metadata(group: ZarrGroup, *, recursive: bool = False) -> dict:
    """Write the consolidated metadata of the hierarchy below the given group.

    Traverse the hierarchy below the given group, collect the metadata of
    each descending node, and write it as a "consolidated_metadata" entry in
    the zarr.json of the group itself. The group's existing metadata is
    preserved; any previous consolidated metadata is replaced. The in-memory
    metadata of the group object is updated accordingly.

    The metadata is collected with a fresh read of the tree from the store
    (not from the possibly stale in-memory tree of the given group), so that
    changes written to the store are picked up rather than clobbered. Only
    the in-memory metadata of the given group object is updated; the nodes
    below it keep the metadata from when they were opened.

    Arguments:
        group : ZarrGroup
            The group to consolidate the metadata for. Must be a group, and
            its store must be writable.
        recursive : bool
            If True (default False), also write consolidated metadata in each
            descending group, each covering the hierarchy below that group
            (with keys relative to it). This makes it possible to open any
            sub-tree from a single request. The top group is written last,
            so that a failure leaves the top zarr.json untouched.

    Returns:
        The consolidated metadata dict that was written for the given group.

    """

    # Checks
    if not isinstance(group, ZarrGroup):
        raise TypeError(
            f"consolidate_metadata() expected a ZarrGroup, got {group.__class__.__name__} {group!r}"
        )
    if not isinstance(group.store, WritableStore):
        raise TypeError(
            f"consolidate_metadata() requires a writable store, got {group.store!r}"
        )
    store = group.store
    path = group.path

    # Read a fresh copy of the tree from the store. This raises (and writes
    # nothing) if the store is inconsistent, and makes sure that we
    # consolidate exactly what is currently in the store. Any existing
    # consolidated metadata is ignored (use_consolidated=False), because
    # it may be stale: the tree is built from the actual zarr.json files.
    fresh_tree = ZarrNode._from_path(store, path, use_consolidated=False)

    # Collect the metadata of all descending nodes, depth-first. The
    # metadata is deep-copied, so that nothing that happens with the
    # collected data can affect the in-memory nodes - and vice versa.
    # A possible consolidated entry on a child group is dropped (from the
    # copy!), because our own entry covers the full hierarchy below the top
    # group, so copying it along would preserve stale data.
    entries = {}  # path -> metadata of the node at that path (without consolidation)
    sub_groups = {}  # path -> node object (descending groups only)

    def collect(node: ZarrGroup, prefix: str) -> None:
        for child in node.children:
            child_path = join(prefix, child.name)
            child_metadata = copy.deepcopy(child.metadata)
            if isinstance(child, ZarrGroup):
                child_metadata.pop("consolidated_metadata", None)
                entries[child_path] = child_metadata
                sub_groups[child_path] = child
                collect(child, child_path)
            else:
                entries[child_path] = child_metadata

    collect(fresh_tree, "")

    # Write the consolidated metadata for each descending group (if
    # requested), with paths relative to that group. Done before the top
    # group, so that a failure leaves the top zarr.json untouched.
    if recursive:
        for sub_path, node in sub_groups.items():
            prefix = sub_path + "/"
            sub_entries = {
                key[len(prefix) :]: meta
                for key, meta in entries.items()
                if key.startswith(prefix)
            }
            _write(node, node, _inline_metadata(sub_entries))

    # Write the consolidated metadata for the top group
    consolidated_metadata = _inline_metadata(entries)
    _write(group, fresh_tree, consolidated_metadata)

    return consolidated_metadata


def _write(
    given_node: ZarrGroup, fresh_node: ZarrGroup, consolidated_metadata: dict
) -> None:
    # Write the zarr.json of the given node, with the consolidated metadata
    # added (replacing any previous version). The in-memory metadata of the
    # node objects is updated, so they stay in sync with the store.
    metadata = dict(fresh_node.metadata)
    metadata.pop("consolidated_metadata", None)
    metadata["consolidated_metadata"] = consolidated_metadata

    json_text = json.dumps(metadata, indent=4)
    fresh_node.store.set(join(fresh_node.path, "zarr.json"), json_text.encode())

    fresh_node._metadata = metadata
    if given_node is not fresh_node:
        given_node._metadata = copy.deepcopy(metadata)
