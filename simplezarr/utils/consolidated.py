# Utilities to write consolidated metadata for a Zarr group, in a format
# compatible with the convention used by zarr-python / xarray.
#
# A Zarr file is a tree of nodes, where each node has its own zarr.json.
# Opening a file (especially over a slow connection) therefore costs one
# store request per node. Consolidated metadata solves this: the metadata of
# all descending nodes is written into the zarr.json of the top group, so
# that the whole tree can be known from a single request.
#
# The consolidated metadata is stored as an entry in the group's metadata::
#
#     "consolidated_metadata": {
#         "kind": "auto",
#         "metadata": {
#             "array1": {...},
#             "sub/array2": {...},
#             ...
#         }
#     }
#
# The keys in ``metadata`` are the paths of the nodes relative to the top
# group (so the top group itself is not included), and the values are the
# full metadata of each node (i.e. the content of its zarr.json).

from __future__ import annotations

import json

from ..nodes import ZarrGroup, join
from ..stores import WritableStore

__all__ = ["consolidate_metadata"]


def consolidate_metadata(group: ZarrGroup, *, recursive: bool = False) -> dict:
    """Write the consolidated metadata of the hierarchy below the given group.

    Traverse the hierarchy below the given group, collect the metadata of
    each descending node, and write it as a "consolidated_metadata" entry in
    the zarr.json of the group itself. The group's existing metadata is
    preserved; any previous consolidated metadata is replaced. The in-memory
    metadata of the group object is updated accordingly.

    Arguments:
        group : ZarrGroup
            The group to consolidate the metadata for. Must be a group, and
            its store must be writable.
        recursive : bool
            If True (default False), also write consolidated metadata in each
            descending group, each covering the hierarchy below that group
            (with keys relative to it). This makes it possible to open any
            sub-tree from a single request.

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

    # Collect the metadata of all descending nodes, depth-first.
    # A possible consolidated entry on a child group is dropped, because
    # our own entry covers the full hierarchy below the top group.
    entries = {}  # path -> metadata of the node at that path (without consolidation)
    sub_groups = {}  # path -> node object (descending groups only)

    def collect(node: ZarrGroup, prefix: str) -> None:
        for child in node.children:
            path = join(prefix, child.name)
            child_metadata = dict(child.metadata)
            if isinstance(child, ZarrGroup):
                child_metadata.pop("consolidated_metadata", None)
                entries[path] = child_metadata
                sub_groups[path] = child
                collect(child, path)
            else:
                entries[path] = child_metadata

    collect(group, "")

    # Write the consolidated metadata for the top group
    consolidated_metadata = {"kind": "auto", "metadata": entries}
    _write(group, consolidated_metadata)

    # And optionally for each descending group, with paths relative to it
    if recursive:
        for path, node in sub_groups.items():
            prefix = path + "/"
            sub_entries = {
                key[len(prefix) :]: meta
                for key, meta in entries.items()
                if key.startswith(prefix)
            }
            _write(node, {"kind": "auto", "metadata": sub_entries})

    return consolidated_metadata


def _write(node: ZarrGroup, consolidated_metadata: dict) -> None:
    # Write the zarr.json of the given node, with the consolidated metadata
    # added (replacing any previous version), and update its in-memory
    # metadata so it stays in sync with the store.
    metadata = dict(node.metadata)
    metadata.pop("consolidated_metadata", None)
    metadata["consolidated_metadata"] = consolidated_metadata

    json_text = json.dumps(metadata, indent=4)
    node.store.set(join(node.path, "zarr.json"), json_text.encode())
    node._metadata = metadata
