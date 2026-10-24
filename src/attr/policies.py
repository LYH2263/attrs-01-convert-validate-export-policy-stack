# SPDX-License-Identifier: MIT

"""
Instance-scoped, thread-local policy stacks.

`attr.validators.set_disabled` is a process-wide, non-thread-safe switch and
field definitions are baked into the class.  The tools in this module instead
manage a stack of frames attached to a *single instance* for the duration of a
*single thread*'s call:

- `ConvertGate` decides when converters run;
- `ValidateGate` decides when validators run, independently of converters and
  without touching the global switch;
- `ExportProfile` decides how `attr.asdict` / `attr.astuple` export.

Frames are pushed with `push` / `pushed` and removed with `pop`.  Every push
must be balanced with a pop; the `pushed` context manager does so even when
the body raises.  An empty stack behaves exactly like the pinned, pre-existing
API.

Pushed frames are never visible to other threads, even for the same instance.
"""

import enum

from contextlib import contextmanager
from typing import Any, Iterator, Optional, TypeVar, Union

from . import _policy
from ._policy import ConvertGate, ExportProfile, Frame, ValidateGate


__all__ = [
    "ConvertGate",
    "ExportProfile",
    "Frame",
    "ValidateGate",
    "pop",
    "push",
    "pushed",
]


_G = TypeVar("_G", bound=enum.Enum)

# Enum members or their string values are accepted at the public boundary.
ConvertGateArg = Union[ConvertGate, str]
ValidateGateArg = Union[ValidateGate, str]
ExportProfileArg = Union[ExportProfile, str]


def _coerce(value: Any, gate_cls: type[_G]) -> Optional[_G]:
    if value is None:
        return None
    if isinstance(value, gate_cls):
        return value

    return gate_cls(value)


def push(
    inst: Any,
    *,
    convert_gate: Optional[ConvertGateArg] = ConvertGate.pin_always,
    validate_gate: Optional[ValidateGateArg] = ValidateGate.respect_global,
    export_profile: Optional[ExportProfileArg] = ExportProfile.pin_full,
) -> Frame:
    """
    Push a policy frame for *inst* onto the current thread's stack and return
    it.

    Every argument defaults to the matching *pin* value, so pushing a frame
    without overrides is a no-op behavior-wise.  Pass `None` for a channel to
    leave the outer frame's value in effect.

    Args:
        inst: Instance of an *attrs*-decorated class.

        convert_gate (ConvertGate | str | None):
            One of `ConvertGate`.

        validate_gate (ValidateGate | str | None):
            One of `ValidateGate`.

        export_profile (ExportProfile | str | None):
            One of `ExportProfile`.

    Returns:
        Frame: The pushed frame; pass it to `pop`.

    Raises:
        ValueError: If a gate value is not a member of its enum.

    ..  warning::

        Every `push` must be paired with a `pop`; prefer the `pushed` context
        manager which pops even when the body raises.
    """
    frame = Frame(
        convert_gate=_coerce(convert_gate, ConvertGate),
        validate_gate=_coerce(validate_gate, ValidateGate),
        export_profile=_coerce(export_profile, ExportProfile),
    )

    return _policy.push(inst, frame)


def pop(inst: Any, frame: Frame) -> None:
    """
    Pop *frame* (returned by `push` or `pushed`) for *inst* from the current
    thread's stack.

    Raises:
        RuntimeError: If nothing is pushed for *inst*, or if *frame* is not
            the current top frame (push/pop must be balanced).
    """
    _policy.pop(inst, frame)


@contextmanager
def pushed(
    inst: Any,
    *,
    convert_gate: Optional[ConvertGateArg] = ConvertGate.pin_always,
    validate_gate: Optional[ValidateGateArg] = ValidateGate.respect_global,
    export_profile: Optional[ExportProfileArg] = ExportProfile.pin_full,
) -> Iterator[Frame]:
    """
    Context manager that pushes a policy frame for *inst* on entry and always
    pops it on exit, including when the body raises.

    See `push` for the arguments.  Nested uses stack naturally: an inner
    frame's non-`None` channels win, and popping it restores the outer frame.
    """
    frame = push(
        inst,
        convert_gate=convert_gate,
        validate_gate=validate_gate,
        export_profile=export_profile,
    )
    try:
        yield frame
    finally:
        _policy.pop(inst, frame)
