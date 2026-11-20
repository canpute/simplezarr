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


def consolidate_metadata(group: ZarrGroup) -> dict:
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

    Returns:
        The consolidated metadata dict that was written.

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

    # Collect the metadata of all descending nodes, depth-first
    def get_meta(node: ZarrGroup, prefix: str) -> dict:
        collected = {}
        for child in node.children:
            path = join(prefix, child.name)
            child_metadata = dict(child.metadata)
            if isinstance(child, ZarrGroup):
                child_metadata.pop("consolidated_metadata", None)
                # Our own consolidated entry covers the full hierarchy below
                # the top group, so a (possibly stale) entry on a child group
                # must not be copied along.
            collected[path] = child_metadata
            if isinstance(child, ZarrGroup):
                collected.update(get_meta(child, path))
        return collected

    consolidated_metadata = {"kind": "auto", "metadata": get_meta(group, "")}

    # Write the zarr.json of the group, with the consolidated metadata added
    metadata = dict(group.metadata)
    metadata.pop("consolidated_metadata", None)  # In case we overwrite a previous version
    metadata["consolidated_metadata"] = consolidated_metadata
    json_text = json.dumps(metadata, indent=4)
    group.store.set(join(group.path, "zarr.json"), json_text.encode())

    # Keep the in-memory node in sync with the store
    group._metadata = metadata

    return consolidated_metadata
