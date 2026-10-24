# SPDX-License-Identifier: MIT

"""
Commonly used hooks for on_setattr.
"""

from . import _policy
from .exceptions import FrozenAttributeError


def pipe(*setters):
    """
    Run all *setters* and return the return value of the last one.

    .. versionadded:: 20.1.0
    """

    def wrapped_pipe(instance, attrib, new_value):
        rv = new_value

        for setter in setters:
            rv = setter(instance, attrib, rv)

        return rv

    return wrapped_pipe


def frozen(_, __, ___):
    """
    Prevent an attribute to be modified.

    .. versionadded:: 20.1.0
    """
    raise FrozenAttributeError


def validate(instance, attrib, new_value):
    """
    Run *attrib*'s validator on *new_value* if it has one.

    Whether validation actually runs is determined by the ``on_setattr`` exit
    of *instance*'s policy stack: ``force_on`` runs it even when the global
    switch disables validators, ``force_off`` skips it without touching the
    global switch, and with an empty stack the global switch is followed.

    .. versionadded:: 20.1.0
    """
    if not _policy.setattr_validates(instance):
        return new_value

    v = attrib.validator
    if not v:
        return new_value

    v(instance, attrib, new_value)

    return new_value


def convert(instance, attrib, new_value):
    """
    Run *attrib*'s converter -- if it has one -- on *new_value* and return the
    result.

    Whether the converter runs is determined by the ``on_setattr`` exit of
    *instance*'s policy stack (``pin_always`` runs it; ``skip`` and
    ``init_only`` skip it here; an empty stack always runs it). This never
    changes whether validation runs.

    .. versionadded:: 20.1.0
    """
    c = attrib.converter
    if c and _policy.setattr_converts(instance):
        # This can be removed once we drop 3.8 and use attrs.Converter instead.
        from ._make import Converter

        if not isinstance(c, Converter):
            return c(new_value)

        return c(new_value, instance, attrib)

    return new_value


# Sentinel for disabling class-wide *on_setattr* hooks for certain attributes.
# Sphinx's autodata stopped working, so the docstring is inlined in the API
# docs.
NO_OP = object()
