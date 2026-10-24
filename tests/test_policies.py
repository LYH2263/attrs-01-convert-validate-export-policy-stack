# SPDX-License-Identifier: MIT

"""
Tests for the instance-scoped, thread-local policy stacks in
`attr.policies` / `attr._policy`.
"""

import threading

import pytest

import attr
import attrs

from attr import _policy, evolve, policies, setters, validators


@attr.s(on_setattr=[setters.convert, setters.validate])
class Hooked:
    x = attr.ib(converter=lambda v: v * 2, validator=validators.instance_of(int))


@attr.s
class Plain:
    x = attr.ib(converter=int, validator=validators.instance_of(int))
    y = attr.ib(default=2)


@attr.s(auto_attribs=True)
class _Inner:
    a: int
    _secret: int = 1


@attr.s(auto_attribs=True)
class _Outer:
    name: str
    inner: object
    _hidden: int = 9


CG = policies.ConvertGate
VG = policies.ValidateGate
EP = policies.ExportProfile


@pytest.fixture(autouse=True)
def _global_validators_enabled():
    # Make sure a stray global disable never leaks between tests.
    validators.set_disabled(False)
    yield
    validators.set_disabled(False)


# ---------------------------------------------------------------------------
# Empty stack == pinned behavior.
# ---------------------------------------------------------------------------


class TestEmptyStackPinned:
    def test_init_converts_and_validates(self):
        c = Plain("3")

        assert c.x == 3
        assert _policy.init_converts(c) is True
        assert _policy.init_validates(c) is True

    def test_init_validation_follows_global_switch(self):
        assert _policy.init_validates(Plain(1)) is True

        c = Plain(1)
        validators.set_disabled(True)
        try:
            assert _policy.init_validates(c) is False
        finally:
            validators.set_disabled(False)

    def test_setattr_gates(self):
        c = Hooked(3)
        assert _policy.setattr_converts(c) is True
        assert _policy.setattr_validates(c) is True

        c.x = 5
        assert c.x == 10

    def test_export_views_are_pin(self):
        o = _Outer("n", _Inner(1))

        view = _policy.asdict_export(o)
        assert view.public_only is False and view.shallow is False
        assert _policy.astuple_export(o) == view

        assert set(attr.asdict(o)) == {"name", "inner", "_hidden"}
        assert attr.astuple(o) == ("n", (1, 1), 9)

    def test_global_disabled_still_converts_everywhere(self):
        validators.set_disabled(True)
        try:
            c = Hooked("1")
            assert c.x == "11"
            c.x = "2"
            assert c.x == "22"
        finally:
            validators.set_disabled(False)


# ---------------------------------------------------------------------------
# on_setattr exit: convert gate.
# ---------------------------------------------------------------------------


class TestSetattrConvertGate:
    def test_skip_suppresses_converter_in_hook(self):
        c = Hooked(1)
        with policies.pushed(c, convert_gate=CG.skip):
            c.x = 7
            assert c.x == 7

    def test_init_only_suppresses_converter_in_hook(self):
        c = Hooked(1)
        with policies.pushed(c, convert_gate=CG.init_only):
            c.x = 7
            assert c.x == 7

    def test_pin_always_converts_in_hook(self):
        c = Hooked(1)
        with policies.pushed(c, convert_gate=CG.pin_always):
            c.x = 7
            assert c.x == 14

    def test_skip_does_not_suppress_validation(self):
        c = Hooked(1)
        with policies.pushed(c, convert_gate=CG.skip):
            with pytest.raises(TypeError):
                c.x = "nope"
        assert c.x == 2

    def test_field_level_on_setattr_also_obeys(self):
        @attr.s
        class C:
            x = attr.ib(
                converter=lambda v: v + 1,
                on_setattr=setters.convert,
            )

        c = C(1)
        assert c.x == 2
        with policies.pushed(c, convert_gate=CG.skip):
            c.x = 40
        assert c.x == 40
        c.x = 1
        assert c.x == 2

    def test_converterless_attribute_unaffected(self):
        @attr.s(on_setattr=[setters.convert, setters.validate])
        class C:
            x = attr.ib(validator=validators.instance_of(int))

        c = C(1)
        with policies.pushed(c, convert_gate=CG.skip):
            c.x = 5
        assert c.x == 5

    def test_frozen_setter_ignores_gates(self):
        @attr.s(on_setattr=setters.frozen)
        class C:
            x = attr.ib()

        c = C(1)
        with policies.pushed(
            c,
            convert_gate=CG.skip,
            validate_gate=VG.force_off,
        ):
            with pytest.raises(attr.exceptions.FrozenAttributeError):
                c.x = 2
        assert c.x == 1

    def test_no_op_bypasses_hooks_entirely(self):
        ran = []

        def hook(inst, attrib, value):
            ran.append(value)
            return value

        @attr.s(on_setattr=hook)
        class C:
            x = attr.ib(on_setattr=setters.NO_OP)
            y = attr.ib()

        c = C(1, 2)
        with policies.pushed(c, validate_gate=VG.force_on):
            c.x = 3
            c.y = 4
        assert ran == [4]


# ---------------------------------------------------------------------------
# on_setattr exit: validate gate.
# ---------------------------------------------------------------------------


class TestSetattrValidateGate:
    def test_respect_global_follows_switch(self):
        c = Hooked(1)
        with policies.pushed(c, validate_gate=VG.respect_global):
            with pytest.raises(TypeError):
                c.x = "x"
        validators.set_disabled(True)
        try:
            with policies.pushed(c, validate_gate=VG.respect_global):
                c.x = "x"
            assert c.x == "xx"
        finally:
            validators.set_disabled(False)

    def test_force_on_validates_while_globally_disabled(self):
        c = Hooked(1)
        validators.set_disabled(True)
        try:
            with policies.pushed(c, validate_gate=VG.force_on):
                with pytest.raises(TypeError):
                    c.x = "x"
            assert c.x == 2  # nothing assigned
        finally:
            validators.set_disabled(False)

    def test_force_off_skips_while_globally_enabled(self):
        c = Hooked(1)
        assert validators.get_disabled() is False
        with policies.pushed(c, validate_gate=VG.force_off):
            c.x = "x"
        assert c.x == "xx"
        assert validators.get_disabled() is False

    def test_force_off_does_not_suppress_converter(self):
        c = Hooked(1)
        with policies.pushed(c, validate_gate=VG.force_off):
            c.x = "3"
        assert c.x == "33"

    def test_force_on_does_not_force_conversion(self):
        @attr.s(on_setattr=[setters.convert, setters.validate])
        class C:
            x = attr.ib(validator=validators.instance_of(int))

        c = C(1)
        with policies.pushed(c, validate_gate=VG.force_on):
            c.x = 6
        assert c.x == 6


# ---------------------------------------------------------------------------
# Construction exit: gates only reachable through evolve (or pre-init push).
# ---------------------------------------------------------------------------


class TestConstructionViaEvolve:
    def test_evolve_pin_still_converts_and_validates(self):
        c = Plain(1)
        e = evolve(c, x="4")
        assert e.x == 4
        assert e is not c

    def test_evolve_skip_does_not_convert(self):
        c = Plain(1)
        with policies.pushed(
            c,
            convert_gate=CG.skip,
            validate_gate=VG.force_off,
        ):
            e = evolve(c, x="42")
        assert e.x == "42"
        assert c.x == 1

    def test_evolve_skip_still_validates(self):
        c = Plain(1)
        with policies.pushed(c, convert_gate=CG.skip):
            with pytest.raises(TypeError):
                evolve(c, x="42")

    def test_evolve_init_only_converts(self):
        c = Plain(1)
        with policies.pushed(c, convert_gate=CG.init_only):
            e = evolve(c, x="42")
        assert e.x == 42

    def test_evolve_force_on_beats_global_disabled(self):
        c = Hooked(1)
        validators.set_disabled(True)
        try:
            with policies.pushed(c, validate_gate=VG.force_on):
                with pytest.raises(TypeError):
                    # Converter runs first ("nope" -> "nopenope"), then the
                    # forced validator rejects the non-int result.
                    evolve(c, x="nope")
        finally:
            validators.set_disabled(False)

    def test_evolve_force_off_skips_init_validation(self):
        c = Plain(1)
        with policies.pushed(
            c,
            convert_gate=CG.skip,
            validate_gate=VG.force_off,
        ):
            e = evolve(c, x=object())
        assert isinstance(e.x, object)

    def test_unchanged_fields_keep_community_default(self):
        # Pin: evolve re-runs the constructor and hence the converter on
        # unchanged fields too -- the policy stack must not change this.
        seen = []

        def conv(v):
            seen.append(v)
            return int(v)

        @attr.s
        class C:
            x = attr.ib(converter=conv)
            y = attr.ib(converter=conv)

        c = C("1", "2")
        seen.clear()
        evolve(c, y="3")
        assert seen == [1, "3"]

        # And under a skip frame both channels skip.
        c = C("1", "2")
        seen.clear()
        with policies.pushed(c, convert_gate=CG.skip):
            e = evolve(c, y="3")
        assert seen == []
        assert e.x == 1 and e.y == "3"

    def test_ambient_is_restored_after_evolve(self):
        c = Plain(1)
        with policies.pushed(c, convert_gate=CG.skip, validate_gate=VG.force_off):
            evolve(c, x="5")
        # The ambient propagation must have been popped: unrelated
        # constructions in this thread are back to pin.
        assert Plain("5").x == 5
        with pytest.raises(TypeError):
            Hooked("nope")


class TestConstructionPreInit:
    def test_frame_pushed_from_pre_init_runs_for_rest_of_init(self):
        @attr.s
        class C:
            x = attr.ib(converter=int, validator=validators.instance_of(int))

            def __attrs_pre_init__(self):
                policies.push(
                    self,
                    convert_gate=CG.skip,
                    validate_gate=VG.force_off,
                )

        c = C("42")
        try:
            assert c.x == "42"
            frames = _policy._state.stacks[id(c)].frames
            assert len(frames) == 1
            policies.pop(c, frames[0])
        finally:
            assert id(c) not in _policy._state.stacks


# ---------------------------------------------------------------------------
# Stack mechanics.
# ---------------------------------------------------------------------------


class TestStackMechanics:
    def test_push_pop_balance(self):
        c = Plain(1)
        frame = policies.push(c, convert_gate=CG.skip)
        assert isinstance(frame, policies.Frame)
        assert _policy.setattr_converts(c) is False
        policies.pop(c, frame)
        assert _policy.setattr_converts(c) is True

    def test_pop_empty_raises(self):
        with pytest.raises(RuntimeError):
            policies.pop(Plain(1), object())

    def test_pop_wrong_instance_raises(self):
        a = Plain(1)
        b = Plain(1)
        frame = policies.push(a, convert_gate=CG.skip)
        with pytest.raises(RuntimeError):
            policies.pop(b, frame)
        policies.pop(a, frame)

    def test_pop_non_top_frame_raises(self):
        c = Plain(1)
        outer = policies.push(c, convert_gate=CG.skip)
        inner = policies.push(c, convert_gate=CG.pin_always)
        with pytest.raises(RuntimeError):
            policies.pop(c, outer)
        policies.pop(c, inner)
        policies.pop(c, outer)

    def test_nested_frames_and_restore(self):
        c = Plain(1)
        with policies.pushed(c, convert_gate=CG.skip):
            assert _policy.setattr_converts(c) is False
            with policies.pushed(
                c,
                convert_gate=CG.pin_always,
                validate_gate=VG.force_off,
            ):
                assert _policy.setattr_converts(c) is True
                assert _policy.setattr_validates(c) is False
            assert _policy.setattr_converts(c) is False
            assert _policy.setattr_validates(c) is True
        assert _policy.setattr_converts(c) is True

    def test_none_channel_falls_through_to_outer(self):
        c = Plain(1)
        with policies.pushed(
            c,
            convert_gate=CG.skip,
            validate_gate=VG.force_off,
        ):
            with policies.pushed(
                c,
                convert_gate=None,
                validate_gate=VG.force_on,
            ):
                assert _policy.setattr_converts(c) is False
                assert _policy.setattr_validates(c) is True

    def test_default_frame_is_pin(self):
        c = Plain(1)
        with policies.pushed(c):
            assert _policy.setattr_converts(c) is True
            assert _policy.setattr_validates(c) is True
            view = _policy.asdict_export(c)
            assert view.public_only is False and view.shallow is False

    def test_string_gate_values_accepted(self):
        c = Plain(1)
        with policies.pushed(
            c,
            convert_gate="skip",
            validate_gate="force_off",
            export_profile="shallow",
        ):
            assert _policy.setattr_converts(c) is False
            assert _policy.setattr_validates(c) is False
            assert _policy.asdict_export(c).shallow is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"convert_gate": "nope"},
            {"validate_gate": "skip"},
            {"export_profile": "force_on"},
        ],
    )
    def test_invalid_gate_value(self, kwargs):
        with pytest.raises(ValueError):
            policies.push(Plain(1), **kwargs)

    def test_exception_pops_clean(self):
        c = Plain(1)
        with pytest.raises(RuntimeError, match="boom"):
            with policies.pushed(c, validate_gate=VG.force_off):
                raise RuntimeError("boom")
        assert id(c) not in _policy._state.stacks
        assert _policy.setattr_validates(c) is True

    def test_frames_scoped_to_instance(self):
        a = Plain(1)
        b = Plain(1)
        with policies.pushed(a, convert_gate=CG.skip):
            assert _policy.setattr_converts(a) is False
            assert _policy.setattr_converts(b) is True

    def test_id_reuse_does_not_confuse_stack(self):
        # A popped entry must be recreated cleanly for a new object that
        # happens to reuse the id.
        c = Plain(1)
        ident = id(c)
        frame = policies.push(c, convert_gate=CG.skip)
        policies.pop(c, frame)
        del c
        d = Plain(1)
        if id(d) == ident:
            assert _policy.setattr_converts(d) is True
        assert _policy.setattr_converts(d) is True


# ---------------------------------------------------------------------------
# Thread isolation.
# ---------------------------------------------------------------------------


class TestThreadIsolation:
    def test_same_instance_different_thread_gates(self):
        c = Hooked(1)
        results = {}
        barrier = threading.Barrier(2)

        def worker(gate, key):
            with policies.pushed(c, convert_gate=gate):
                barrier.wait()
                c.x = 7
                results[key] = c.x

        t1 = threading.Thread(
            target=worker, args=(CG.skip, "skip")
        )
        t2 = threading.Thread(
            target=worker, args=(CG.pin_always, "pin")
        )
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["skip"] == 7
        assert results["pin"] == 14
        assert _policy._own_frames(c) == ()

    def test_frame_invisible_to_other_thread(self):
        c = Hooked(1)
        entered = threading.Event()
        release = threading.Event()

        def worker():
            with policies.pushed(c, convert_gate=CG.skip):
                entered.set()
                release.wait(5)

        t = threading.Thread(target=worker)
        t.start()
        entered.wait(5)
        try:
            # Main thread keeps pin conversion despite the child's frame.
            c.x = 3
            assert c.x == 6
        finally:
            release.set()
            t.join()

    def test_export_frame_invisible_to_other_thread(self):
        o = _Outer("n", _Inner(1))
        seen = {}
        entered = threading.Event()
        release = threading.Event()

        def worker():
            with policies.pushed(o, export_profile=EP.public_only):
                entered.set()
                release.wait(5)
                seen["worker"] = set(attr.asdict(o))

        t = threading.Thread(target=worker)
        t.start()
        entered.wait(5)
        try:
            seen["main"] = set(attr.asdict(o))
        finally:
            release.set()
            t.join()
        assert seen["main"] == {"name", "inner", "_hidden"}
        assert seen["worker"] == {"name", "inner"}

    def test_evolve_ambient_does_not_leak_to_other_thread(self):
        c = Plain(1)
        ambient_seen = []

        def worker():
            # Ambient list is thread local: the main thread's evolve
            # propagation is invisible here.
            ambient_seen.append(list(_policy._state.ambient))

        with policies.pushed(c, convert_gate=CG.skip):
            t = threading.Thread(target=worker)
            t.start()
            t.join()
            evolve(c, x=1)
        assert ambient_seen == [[]]


# ---------------------------------------------------------------------------
# asdict exit.
# ---------------------------------------------------------------------------


def _deep():
    return _Outer(
        "n",
        {"k": [_Inner(1)], "t": (_Inner(2),)},
    )


class TestAsdictProfile:
    def test_public_only_top_and_nested(self):
        o = _Outer("n", _Inner(1))
        with policies.pushed(o, export_profile=EP.public_only):
            d = attr.asdict(o)
        assert set(d) == {"name", "inner"}
        assert set(d["inner"]) == {"a"}

    def test_public_only_inside_collections(self):
        o = _deep()
        with policies.pushed(o, export_profile=EP.public_only):
            d = attr.asdict(o)
        assert set(d["inner"]["k"][0]) == {"a"}
        assert set(d["inner"]["t"][0]) == {"a"}

    def test_public_only_with_filter_and_serializer(self):
        o = _Outer("n", _Inner(1))
        filtered = []

        def flt(a, value):
            filtered.append(a.name)
            return a.name != "name"

        with policies.pushed(o, export_profile=EP.public_only):
            d = attr.asdict(
                o,
                filter=flt,
                value_serializer=lambda i, a, v: f"S:{v!r}",
            )
        assert set(d) == {"inner"}
        assert isinstance(d["inner"], str)
        # The serializer turned the nested instance into a string before
        # recursion, so the filter never saw its fields.
        assert "_hidden" not in filtered
        assert set(filtered) == {"name", "inner"}

    def test_shallow_keeps_nested_attrs_raw(self):
        inner = _Inner(1)
        o = _Outer("n", inner)
        with policies.pushed(o, export_profile=EP.shallow):
            d = attr.asdict(o)
        assert d["inner"] is inner
        assert set(d) == {"name", "inner", "_hidden"}

    def test_shallow_inside_collections(self):
        o = _deep()
        with policies.pushed(o, export_profile=EP.shallow):
            d = attr.asdict(o, retain_collection_types=True)
        assert d["inner"]["k"] == [_Inner(1)]
        assert d["inner"]["t"] == (_Inner(2),)
        # The surrounding dict/list/tuple containers are still rebuilt.
        assert d["inner"] is not o.inner

    def test_shallow_with_value_serializer(self):
        o = _Outer("n", _Inner(1))
        marker = object()

        def serializer(inst, a, v):
            if a is not None and a.name == "inner":
                return marker
            return v

        with policies.pushed(o, export_profile=EP.shallow):
            d = attr.asdict(o, value_serializer=serializer)
        # Serializer runs first (after the filter), and shallow then keeps
        # its opaque result instead of recursing.
        assert d["inner"] is marker

    def test_dict_factory_retained(self):
        from collections import OrderedDict

        o = _Outer("n", _Inner(1))
        with policies.pushed(o, export_profile=EP.public_only):
            d = attr.asdict(o, dict_factory=OrderedDict)
        assert isinstance(d, OrderedDict)
        assert isinstance(d["inner"], OrderedDict)

    def test_recurse_false_with_public_only(self):
        o = _Outer("n", _Inner(1))
        with policies.pushed(o, export_profile=EP.public_only):
            d = attr.asdict(o, recurse=False)
        assert set(d) == {"name", "inner"}
        assert d["inner"] is o.inner

    def test_nested_instance_own_stack_ignored_during_recursion(self):
        inner = _Inner(1)
        outer = _Outer("n", inner)
        # The export is governed by the entry instance; the child's own
        # frame does not interrupt the carried profile.
        with policies.pushed(outer, export_profile=EP.public_only):
            with policies.pushed(inner, export_profile=EP.pin_full):
                d = attr.asdict(outer)
        assert set(d["inner"]) == {"a"}

    def test_explicit_nested_call_reads_its_own_stack(self):
        inner = _Inner(1)
        outer = _Outer("n", inner)
        with policies.pushed(outer, export_profile=EP.shallow):
            d = attr.asdict(outer)
            assert d["inner"] is inner
            # An explicit top-level call on the nested instance is pin.
            assert attr.asdict(inner) == {"a": 1, "_secret": 1}


# ---------------------------------------------------------------------------
# astuple exit.
# ---------------------------------------------------------------------------


class TestAstupleProfile:
    def test_public_only(self):
        o = _Outer("n", _Inner(1))
        with policies.pushed(o, export_profile=EP.public_only):
            assert attr.astuple(o) == ("n", (1,))

    def test_public_only_with_filter_and_factory(self):
        o = _Outer("n", _Inner(1))
        with policies.pushed(o, export_profile=EP.public_only):
            t = attr.astuple(
                o,
                filter=lambda a, v: a.name != "name",
                tuple_factory=list,
            )
        # The list factory propagates into nested exports (pin semantics).
        assert t == [[1]]

    def test_shallow(self):
        inner = _Inner(1)
        o = _Outer("n", inner)
        with policies.pushed(o, export_profile=EP.shallow):
            t = attr.astuple(o)
        assert t[1] is inner

    def test_shallow_inside_collections(self):
        o = _deep()
        with policies.pushed(o, export_profile=EP.shallow):
            t = attr.astuple(o, retain_collection_types=True)
        # containers rebuilt, attrs members left raw
        assert t[1]["k"] == [_Inner(1)]
        assert t[1]["t"] == (_Inner(2),)

    def test_retain_collection_types_default_semantics(self):
        o = _Outer("n", [_Inner(1), _Inner(2)])
        # pin: nested attrs members convert, outer list retained anyway
        pin = attr.astuple(o)
        assert pin[1] == [(1, 1), (2, 1)]
        with policies.pushed(o, export_profile=EP.shallow):
            t = attr.astuple(o)
        assert t[1] == [_Inner(1), _Inner(2)]


# ---------------------------------------------------------------------------
# attrs namespace keeps retain_collection_types=True default and profile.
# ---------------------------------------------------------------------------


def test_attrs_namespace_exports_read_profile():
    o = _Outer("n", (1, 2))
    # attrs.astuple retains the tuple type; profile still applies.
    pin = attrs.astuple(o)
    assert pin[1] == (1, 2)
    with policies.pushed(o, export_profile=EP.public_only):
        assert attrs.astuple(o) == ("n", (1, 2))
    d = attrs.asdict(o)
    with policies.pushed(o, export_profile=EP.public_only):
        d2 = attrs.asdict(o)
    assert set(d) == {"name", "inner", "_hidden"}
    assert set(d2) == {"name", "inner"}


# ---------------------------------------------------------------------------
# Frozen classes and global switch integrity.
# ---------------------------------------------------------------------------


def test_frozen_class_permanent_freeze_and_export():
    @attr.frozen
    class F:
        x: int = 1
        _y: int = 2

    f = F()
    with policies.pushed(
        f,
        convert_gate=CG.skip,
        validate_gate=VG.force_off,
    ):
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            f.x = 3
        d = attr.asdict(f)
    assert d == {"x": 1, "_y": 2}
    g = evolve(f, x=4)
    assert g.x == 4


def test_global_switch_never_mutated_by_frames():
    c = Plain(1)
    assert validators.get_disabled() is False
    with policies.pushed(c, validate_gate=VG.force_off):
        pass
    assert validators.get_disabled() is False
    validators.set_disabled(True)
    try:
        with policies.pushed(c, validate_gate=VG.force_on):
            pass
        assert validators.get_disabled() is True
    finally:
        validators.set_disabled(False)


def test_init_does_not_run_on_setattr_with_frame_pushed_elsewhere():
    calls = []

    def hook(inst, attrib, value):
        calls.append(value)
        return value

    @attr.s(on_setattr=hook)
    class C:
        x = attr.ib()

    holder = C(1)
    # A frame on another instance must not make __init__ invoke hooks.
    with policies.pushed(holder, convert_gate=CG.skip):
        C(2)
    assert calls == []


def test_no_new_required_constructor_arguments():
    # Plain construction signatures are unchanged (no policy kwargs on init).
    import inspect

    sig = inspect.signature(Hooked.__init__)
    assert list(sig.parameters) == ["self", "x"]
