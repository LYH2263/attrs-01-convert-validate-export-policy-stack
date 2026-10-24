# SPDX-License-Identifier: MIT

"""
Per-instance, thread-local policy stacks for conversion, validation, and
exporting.

This module is the private engine behind `attr.policies`.  Unlike
`attr.validators.set_disabled`, the stacks are scoped to one instance and one
thread: frames pushed in one thread are never visible in another.

Each of the four *exit points* has its own effective-view reader:

- construction (the generated ``__init__``): `init_converts` /
  `init_validates`;
- ``on_setattr`` hooks (`attr.setters.pipe` / ``setters.convert`` /
  ``setters.validate``): `setattr_converts` / `setattr_validates`;
- `attr.asdict`: `asdict_export`;
- `attr.astuple`: `astuple_export`.

They deliberately do not share a single monolithic ``get_effective`` helper,
because merging with the global validator switch and with field definitions is
specific to each exit.

Frames may additionally become visible during construction through the
*ambient* chain, which `evolve` uses to propagate the source instance's stack
to the new instance's ``__init__``.
"""

import enum
import threading
import weakref
from typing import Any, Callable, Iterator, Literal, Optional, TypeVar

from . import _config


class ConvertGate(enum.Enum):
    """
    Effective converter timing for one policy frame.

    Attributes:
        pin_always:
            Keep the pinned behavior: run the converter during construction
            and from the ``on_setattr`` convert hook.
        skip:
            Skip the converter in the current frame, both during construction
            and from the ``on_setattr`` convert hook. Validation is *not*
            affected.
        init_only:
            Run the converter during construction, but skip it in the
            ``on_setattr`` convert hook.
    """

    pin_always = "pin_always"
    skip = "skip"
    init_only = "init_only"


class ValidateGate(enum.Enum):
    """
    Effective validator execution for one policy frame.

    Attributes:
        respect_global:
            Follow the global switch (`attr.validators.get_disabled` /
            `attr.get_run_validators`).
        force_on:
            Run validators in the current frame even if the global switch has
            them disabled. The global switch is not modified.
        force_off:
            Skip validators in the current frame without touching the global
            switch. Conversion is *not* affected.
    """

    respect_global = "respect_global"
    force_on = "force_on"
    force_off = "force_off"


class ExportProfile(enum.Enum):
    """
    Effective export shape for one policy frame, read by `attr.asdict` and
    `attr.astuple`.

    Attributes:
        pin_full:
            Keep the pinned behavior of ``asdict``/``astuple`` and their
            *recurse*, *filter*, factories, *retain_collection_types* and
            *value_serializer* arguments.
        public_only:
            Drop fields whose name starts with an underscore while exporting.
        shallow:
            Do not recurse into nested *attrs* instances; leave them as-is
            (collections are still processed according to
            *retain_collection_types*).
    """

    pin_full = "pin_full"
    public_only = "public_only"
    shallow = "shallow"


_Gate = TypeVar("_Gate", bound=enum.Enum)
_ChannelName = Literal["convert_gate", "validate_gate", "export_profile"]


class Frame:
    """
    One immutable entry of a policy stack.

    Each channel may be `None`, which means "no opinion": resolution falls
    through to the frame below (and ultimately to the pinned defaults).

    Do not instantiate directly -- use `attr.policies.push` /
    `attr.policies.pushed`.
    """

    __slots__ = ("convert_gate", "export_profile", "validate_gate")

    def __init__(
        self,
        convert_gate: Optional[ConvertGate],
        validate_gate: Optional[ValidateGate],
        export_profile: Optional[ExportProfile],
    ) -> None:
        self.convert_gate = convert_gate
        self.validate_gate = validate_gate
        self.export_profile = export_profile

    def __repr__(self) -> str:
        return (
            f"Frame(convert_gate={self.convert_gate!r}, "
            f"validate_gate={self.validate_gate!r}, "
            f"export_profile={self.export_profile!r})"
        )


class _Entry:
    """
    The per-thread stack of one instance, plus a (preferably weak) reference
    to it for identity checks.
    """

    __slots__ = ("frames", "ref")

    def __init__(self, inst: Any) -> None:
        self.frames: list[Frame] = []
        self.ref: Callable[[], Any]
        try:
            self.ref = weakref.ref(inst)
        except TypeError:
            # Instances without weakref support (e.g. slotted classes created
            # with weakref_slot=False) hold the entry for the lifetime of the
            # active frame only -- the entry is deleted as soon as its stack
            # becomes empty again.
            self.ref = lambda: inst

    def resolve(self) -> Any:
        return self.ref()


class _State(threading.local):
    def __init__(self) -> None:
        super().__init__()
        # id(instance) -> _Entry. Entries only exist while at least one frame
        # is pushed in this thread.
        self.stacks: dict[int, _Entry] = {}
        # Frame lists inherited by constructions initiated from other
        # instances (evolve). Innermost propagation last.
        self.ambient: list[list[Frame]] = []


_state = _State()


def _own_frames(inst: Any) -> tuple[Frame, ...]:
    """
    Return a snapshot of this thread's frames for *inst*, top frame first.
    """
    entry = _state.stacks.get(id(inst))
    if entry is None or entry.resolve() is not inst:
        return ()

    return tuple(reversed(entry.frames))


def _ambient_frames() -> list[Frame]:
    """
    Return the propagated construction frames, innermost propagation first
    and each of them top frame first.
    """
    frames: list[Frame] = []
    for stack in reversed(_state.ambient):
        frames.extend(reversed(tuple(stack)))

    return frames


def _iter_effective_frames(inst: Any) -> Iterator[Frame]:
    # The instance's own frames take precedence over propagated construction
    # frames.
    yield from _own_frames(inst)
    yield from _ambient_frames()


def _effective(inst: Any, channel: _ChannelName, default: _Gate) -> _Gate:
    """
    Walk the frames visible to *inst* and return the first non-`None` value
    for *channel*; fall back to *default*.
    """
    for frame in _iter_effective_frames(inst):
        value = getattr(frame, channel)
        if value is not None:
            return value

    return default


# ---------------------------------------------------------------------------
# Forked effective-view readers. One per exit; do not merge.
# ---------------------------------------------------------------------------


def init_converts(inst: Any) -> bool:
    """
    Construction exit: whether converters run while *inst* is initialized.

    `ConvertGate.skip` is the only gate that suppresses conversion here;
    ``init_only`` still converts during construction. An empty stack always
    converts, matching the pinned behavior.
    """
    return (
        _effective(inst, "convert_gate", ConvertGate.pin_always)
        is not ConvertGate.skip
    )


def init_validates(inst: Any) -> bool:
    """
    Construction exit: whether validators run after *inst* is initialized.

    `ValidateGate.force_on` runs them even when the global switch disables
    validation; `ValidateGate.force_off` skips them without touching the
    global switch; ``respect_global`` and an empty stack follow the global
    switch.
    """
    gate = _effective(inst, "validate_gate", ValidateGate.respect_global)
    if gate is ValidateGate.force_on:
        return True
    if gate is ValidateGate.force_off:
        return False

    return _config._run_validators


def setattr_converts(inst: Any) -> bool:
    """
    ``on_setattr`` exit: whether the convert hook runs on *inst*.

    Only `ConvertGate.pin_always` converts here -- both ``skip`` and
    ``init_only`` suppress the converter in assignment hooks. An empty stack
    always converts, matching the pinned behavior.
    """
    return (
        _effective(inst, "convert_gate", ConvertGate.pin_always)
        is ConvertGate.pin_always
    )


def setattr_validates(inst: Any) -> bool:
    """
    ``on_setattr`` exit: whether the validate hook runs on *inst*.

    See `init_validates` for gate semantics; this reader is intentionally
    separate because it belongs to a different exit (``on_setattr`` /
    ``setters.pipe`` instead of construction).
    """
    gate = _effective(inst, "validate_gate", ValidateGate.respect_global)
    if gate is ValidateGate.force_on:
        return True
    if gate is ValidateGate.force_off:
        return False

    return _config._run_validators


class ExportView:
    """
    The merged export view consumed by one of the two export exits.

    Attributes:
        public_only: Drop fields whose name starts with an underscore.
        shallow: Leave nested *attrs* instances unrecursed.
    """

    __slots__ = ("public_only", "shallow")

    def __init__(self, public_only: bool, shallow: bool) -> None:
        self.public_only = public_only
        self.shallow = shallow

    def __repr__(self) -> str:
        return (
            f"ExportView(public_only={self.public_only!r}, "
            f"shallow={self.shallow!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExportView):
            return NotImplemented
        return (
            self.public_only == other.public_only
            and self.shallow == other.shallow
        )


_PIN_EXPORT_VIEW = ExportView(public_only=False, shallow=False)
_PUBLIC_ONLY_EXPORT_VIEW = ExportView(public_only=True, shallow=False)
_SHALLOW_EXPORT_VIEW = ExportView(public_only=False, shallow=True)


def _export_view(inst: Any) -> ExportView:
    profile = _effective(inst, "export_profile", ExportProfile.pin_full)
    if profile is ExportProfile.public_only:
        return _PUBLIC_ONLY_EXPORT_VIEW
    if profile is ExportProfile.shallow:
        return _SHALLOW_EXPORT_VIEW

    return _PIN_EXPORT_VIEW


def asdict_export(inst: Any) -> ExportView:
    """
    ``asdict`` exit: the merged `ExportProfile` view for *inst*.

    The *filter*, *dict_factory*, *retain_collection_types* and
    *value_serializer* arguments keep their existing semantics; only the
    profile is resolved here.
    """
    return _export_view(inst)


def astuple_export(inst: Any) -> ExportView:
    """
    ``astuple`` exit: the merged `ExportProfile` view for *inst*.

    Deliberately separate from `asdict_export`: the two exports keep their
    own factories and *retain_collection_types* semantics.
    """
    return _export_view(inst)


# ---------------------------------------------------------------------------
# Stack mutation and construction propagation.
# ---------------------------------------------------------------------------


def push(inst: Any, frame: Frame) -> Frame:
    """
    Push an already-normalized *frame* onto this thread's stack for *inst*.
    """
    state = _state
    key = id(inst)
    entry = state.stacks.get(key)
    if entry is None or entry.resolve() is not inst:
        # Either there never was an entry, or a dead object's id got reused.
        entry = _Entry(inst)
        state.stacks[key] = entry

    entry.frames.append(frame)

    return frame


def pop(inst: Any, frame: Frame) -> None:
    """
    Pop *frame* from this thread's stack for *inst*.

    Raises:
        RuntimeError: If no frame is pushed for *inst*, or if *frame* is not
            the current top frame (an unbalanced push/pop).
    """
    state = _state
    key = id(inst)
    entry = state.stacks.get(key)
    if entry is None or entry.resolve() is not inst or not entry.frames:
        msg = "No policy frame is pushed for this instance in this thread."
        raise RuntimeError(msg)

    if entry.frames[-1] is not frame:
        msg = (
            "Cannot pop a policy frame that is not the current top frame; "
            "push and pop must be balanced."
        )
        raise RuntimeError(msg)

    entry.frames.pop()
    if not entry.frames:
        del state.stacks[key]


def begin_construction(source: Any) -> bool:
    """
    Make *source*'s current frames visible to a construction about to happen
    in this thread (used by `evolve`).

    Returns `True` if anything was propagated and must be balanced with
    `end_construction`.
    """
    entry = _state.stacks.get(id(source))
    if entry is None or entry.resolve() is not source or not entry.frames:
        return False

    _state.ambient.append(entry.frames)

    return True


def end_construction(source: Any) -> None:
    """
    Undo one `begin_construction` for *source*.
    """
    entry = _state.stacks.get(id(source))
    frames = (
        entry.frames
        if entry is not None and entry.resolve() is source
        else None
    )
    if (
        _state.ambient
        and frames is not None
        and _state.ambient[-1] is frames
    ):
        _state.ambient.pop()
        return

    msg = "Unbalanced construction policy propagation."
    raise RuntimeError(msg)
