"""Fixtures comunes: la red queda bloqueada en todos los tests (cero red)."""
import socket

import pytest


@pytest.fixture(autouse=True)
def sin_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cualquier intento de conexión falla."""

    def bloqueado(*args, **kwargs):
        raise RuntimeError("Los tests no pueden usar la red")

    monkeypatch.setattr(socket.socket, "connect", bloqueado)
    monkeypatch.setattr(socket, "getaddrinfo", bloqueado)
