# -*- coding: utf-8 -*-
r"""Remote access in granddb: known hosts only, names kept as data.

- An SSH data source used to accept any host key, so whoever answered in the
  server's place received the credentials.
- File names from the database reached the remote shell unquoted.
- Names with a path in them placed downloads outside the incoming directory.
"""

import io

import pytest

pytest.importorskip("paramiko")
pytest.importorskip("scp")

import paramiko  # noqa: E402

from granddb import datamanager as dm  # noqa: E402


def _ssh_source(tmp_path):
    source = dm.Datasource("repo", "ssh", "example.invalid", 22, ["/data"], str(tmp_path) + "/")
    assert isinstance(source, dm.DatasourceSsh)
    return source


class _FakeClient:
    r"""Records remote commands; finds nothing."""

    def __init__(self):
        self.commands = []

    def exec_command(self, command):
        self.commands.append(command)
        return io.StringIO(), io.StringIO(), io.StringIO()


def test_unknown_host_keys_are_rejected(tmp_path, monkeypatch):
    policies = []
    monkeypatch.setattr(paramiko.SSHClient, "set_missing_host_key_policy",
                        lambda self, policy: policies.append(policy))
    monkeypatch.setattr(paramiko.SSHClient, "load_system_host_keys", lambda self: None)

    def refuse(self, **kwargs):
        raise paramiko.SSHException("Server 'example.invalid' not found in known_hosts")

    monkeypatch.setattr(paramiko.SSHClient, "connect", refuse)
    _ssh_source(tmp_path).set_client(max_retries=1)
    assert policies and all(isinstance(p, paramiko.RejectPolicy) for p in policies)


def test_remote_commands_quote_the_names(tmp_path):
    client = _FakeClient()
    source = _ssh_source(tmp_path)
    source._get_file(client, "/data/", "a b.root")
    source.get_dir(client, "/data/", "run 1")
    assert client.commands == ["find /data/ -type f -name 'a b.root'",
                               "find /data/ -type d -name 'run 1'"]


@pytest.mark.parametrize("name", ["../escape.root", "/etc/passwd", "sub/dir.root", "..", ""])
def test_names_with_a_path_are_refused(tmp_path, name):
    client = _FakeClient()
    source = _ssh_source(tmp_path)
    assert source._get_file(client, "/data/", name) is None
    assert source.get_dir(client, "/data/", name) is None
    assert client.commands == []
    http = dm.Datasource("web", "http", "example.invalid", 80, ["/data"], str(tmp_path) + "/")
    assert http._get_file("http://example.invalid/data/x", name) is None
