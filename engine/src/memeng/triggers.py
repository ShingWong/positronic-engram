# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Trigger bus — inline handlers + always-complete trace."""
from __future__ import annotations

from .models import Handler, Handlers, Trigger


class TriggerBus:
    def __init__(self) -> None:
        self.trace: list[Trigger] = []

    def emit(self, type_: str, tau: float, **payload: object) -> None:
        t = Trigger(type=type_, tau=tau, payload=payload)
        self.trace.append(t)

    def deliver(self, type_: str, handlers: Handlers | None) -> None:
        """Invoke registered handler(s) for already-emitted triggers."""
        if not handlers:
            return
        for t in self.trace:
            if t.type == type_:
                h: Handler | None = handlers.get(type_)
                if h:
                    h(t)
