import re
from collections.abc import Sequence
from typing import Any


def resolve_matching_names(
    keys: str | Sequence[str],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
    """Resolve regex keys against names with IsaacLab-compatible ordering."""
    if isinstance(keys, str):
        keys = [keys]

    index_list: list[int] = []
    names_list: list[str] = []
    key_idx_list: list[int] = []
    target_matches: list[str | None] = [None for _ in list_of_strings]
    key_matches: list[list[str]] = [[] for _ in keys]

    for target_index, candidate in enumerate(list_of_strings):
        for key_index, regex in enumerate(keys):
            if not re.fullmatch(regex, candidate):
                continue
            if target_matches[target_index] is not None:
                raise ValueError(
                    f"Multiple matches for '{candidate}': "
                    f"'{target_matches[target_index]}' and '{regex}'!"
                )
            target_matches[target_index] = regex
            index_list.append(target_index)
            names_list.append(candidate)
            key_idx_list.append(key_index)
            key_matches[key_index].append(candidate)

    _raise_on_unmatched(keys, key_matches, list_of_strings)

    if preserve_order:
        index_list, names_list = _reorder_by_key(index_list, names_list, key_idx_list)

    return index_list, names_list


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str], list[Any]]:
    """Resolve regex-value mappings against names with IsaacLab-compatible ordering."""
    if not isinstance(data, dict):
        raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")

    index_list: list[int] = []
    names_list: list[str] = []
    values_list: list[Any] = []
    key_idx_list: list[int] = []
    target_matches: list[str | None] = [None for _ in list_of_strings]
    key_matches: list[list[str]] = [[] for _ in data]

    for target_index, candidate in enumerate(list_of_strings):
        for key_index, (regex, value) in enumerate(data.items()):
            if not re.fullmatch(regex, candidate):
                continue
            if target_matches[target_index] is not None:
                raise ValueError(
                    f"Multiple matches for '{candidate}': "
                    f"'{target_matches[target_index]}' and '{regex}'!"
                )
            target_matches[target_index] = regex
            index_list.append(target_index)
            names_list.append(candidate)
            values_list.append(value)
            key_idx_list.append(key_index)
            key_matches[key_index].append(candidate)

    _raise_on_unmatched(data.keys(), key_matches, list_of_strings)

    if preserve_order:
        index_list, names_list, values_list = _reorder_by_key(
            index_list,
            names_list,
            key_idx_list,
            values_list,
        )

    return index_list, names_list, values_list


def _raise_on_unmatched(keys, key_matches: Sequence[Sequence[str]], list_of_strings: Sequence[str]) -> None:
    if all(key_matches):
        return
    msg = "\n"
    for key, value in zip(keys, key_matches):
        msg += f"\t{key}: {list(value)}\n"
    msg += f"Available strings: {list(list_of_strings)}\n"
    raise ValueError(f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}")


def _reorder_by_key(index_list, names_list, key_idx_list, values_list=None):
    order = sorted(range(len(index_list)), key=lambda i: (key_idx_list[i], i))
    indices = [index_list[i] for i in order]
    names = [names_list[i] for i in order]
    if values_list is None:
        return indices, names
    values = [values_list[i] for i in order]
    return indices, names, values
