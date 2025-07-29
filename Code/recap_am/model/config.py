import collections
import typing as t
from pathlib import Path

import tomlkit as toml
from packaging import version

#j'ai remplacé dans class config, l'argument collections.MutableMapping par collections.abc.MutableMapping car 
#  MutableMapping = _alias(collections.abc.MutableMapping, 2) et je n'ai pls eu d'erreur dans mon fichienotebook avec ca lorsque
#je testais la class config 

#class Config(collections.MutableMapping):
class Config(collections.abc.MutableMapping):
    _instance = None
    _store: t.MutableMapping[str, t.Any]
    #_file = Path("config.toml")
    #_template = Path("config-example.toml")
    _file = Path(__file__).resolve().parent.parent.parent / "config.toml"
    _template = Path(__file__).resolve().parent.parent.parent / "config-example.toml"
    @classmethod
    def get_instance(cls):
        """ Static access method. """
        if cls._instance is None:
            cls()
        return cls._instance

    def __init__(self):
        """ Private constructor."""
        if Config._instance is not None:
            raise RuntimeError("This class is a singleton!")
        else:
            Config._instance = self
            with self._file.open() as f:
                self._store = toml.parse(f.read())

            with self._template.open() as f:
                template = toml.parse(f.read())

            if "version" not in self:
                raise RuntimeError(
                    f"The attribute 'version' is missing in '{self._file}'."
                )

            if "version" not in template:
                raise RuntimeError(
                    f"The attribute 'version' is missing in '{self._template}'."
                )

            self_version = version.parse(str(self["version"]))
            template_version = version.parse(str(template["version"]))

            if self_version != template_version:
                raise RuntimeError(
                    f"The version of '{self._file}' ({self_version}) is not equal to "
                    f"the version of '{self._template}' ({template_version}). "
                    "Please update your config!"
                )

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return repr(self._store)

    def __str__(self):
        return str(self._store)


config = Config.get_instance()
