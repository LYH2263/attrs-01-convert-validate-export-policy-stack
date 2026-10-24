import enum
from typing import ContextManager, Literal

from attrs import AttrsInstance

class ConvertGate(enum.Enum):
    pin_always = "pin_always"
    skip = "skip"
    init_only = "init_only"

class ValidateGate(enum.Enum):
    respect_global = "respect_global"
    force_on = "force_on"
    force_off = "force_off"

class ExportProfile(enum.Enum):
    pin_full = "pin_full"
    public_only = "public_only"
    shallow = "shallow"

class Frame:
    convert_gate: ConvertGate | None
    validate_gate: ValidateGate | None
    export_profile: ExportProfile | None

_ConvertGateArg = ConvertGate | Literal["pin_always", "skip", "init_only"]
_ValidateGateArg = ValidateGate | Literal[
    "respect_global", "force_on", "force_off"
]
_ExportProfileArg = ExportProfile | Literal[
    "pin_full", "public_only", "shallow"
]

def push(
    inst: AttrsInstance,
    *,
    convert_gate: _ConvertGateArg | None = ...,
    validate_gate: _ValidateGateArg | None = ...,
    export_profile: _ExportProfileArg | None = ...,
) -> Frame: ...
def pop(inst: AttrsInstance, frame: Frame) -> None: ...
def pushed(
    inst: AttrsInstance,
    *,
    convert_gate: _ConvertGateArg | None = ...,
    validate_gate: _ValidateGateArg | None = ...,
    export_profile: _ExportProfileArg | None = ...,
) -> ContextManager[Frame]: ...
