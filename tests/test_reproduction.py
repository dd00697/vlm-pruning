from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_reproduce():
    path = ROOT / "scripts/reproduce.py"
    spec = importlib.util.spec_from_file_location("reproduce", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registry_expands_only_existing_python_entrypoints() -> None:
    module = load_reproduce()
    registry = module.load_registry()
    context = {
        "python": "python",
        "max_workers": 1,
        "ivc_python": "python3.10",
        "ivc_source": "third_party/IVC-Prune",
        "ivc_model": "snapshot",
    }
    commands = module.expand(registry, "all", context)
    assert commands
    for _, command in commands:
        if len(command) > 1 and command[1].endswith(".py"):
            assert (ROOT / command[1]).is_file(), command[1]


def test_verify_pipeline_is_inference_free() -> None:
    module = load_reproduce()
    registry = module.load_registry()
    context = {
        "python": "python",
        "max_workers": 1,
        "ivc_python": "python3.10",
        "ivc_source": "third_party/IVC-Prune",
        "ivc_model": "snapshot",
    }
    commands = module.expand(registry, "verify", context)
    assert [Path(command[1]).name for _, command in commands] == [
        "generate_readme_figures.py",
        "verify_dissertation_results.py",
    ]
