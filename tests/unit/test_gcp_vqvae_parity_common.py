import ctypes
import ctypes.util
import json
from types import SimpleNamespace

import pytest
import torch

import scripts.gcp_vqvae_parity.common as common
from scripts.gcp_vqvae_parity.common import (
    atomic_write_json,
    capture_environment,
    configure_determinism,
    discover_inputs,
    require_cuda,
    select_shard,
    sha256_file,
)
from scripts.gcp_vqvae_parity.preprocessing import build_preprocessing_audit


def test_discover_inputs_is_sorted_and_hashed(tmp_path):
    (tmp_path / "b.cif").write_text("beta")
    (tmp_path / "a.cif").write_text("alpha")
    (tmp_path / "ignored.txt").write_text("ignored")

    records = discover_inputs(tmp_path)

    assert [record.name for record in records] == ["a.cif", "b.cif"]
    assert records[0].sha256 == sha256_file(tmp_path / "a.cif")
    assert len(records[0].sha256) == 64


def test_discover_inputs_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No structure inputs"):
        discover_inputs(tmp_path)


def test_select_shard_partitions_without_overlap(tmp_path):
    for index in range(7):
        (tmp_path / f"{index}.cif").write_text(str(index))
    records = discover_inputs(tmp_path)
    shards = [select_shard(records, shard_index=i, num_shards=3) for i in range(3)]
    names = [[record.name for record in shard] for shard in shards]
    assert names == [["0.cif", "3.cif", "6.cif"], ["1.cif", "4.cif"], ["2.cif", "5.cif"]]
    assert sorted(name for shard in names for name in shard) == [record.name for record in records]


def test_select_shard_validates_indices(tmp_path):
    (tmp_path / "a.cif").write_text("alpha")
    with pytest.raises(ValueError, match="shard_index"):
        select_shard(discover_inputs(tmp_path), shard_index=2, num_shards=2)


def test_select_shard_rejects_invalid_shard_count():
    with pytest.raises(ValueError, match="num_shards"):
        select_shard([], shard_index=0, num_shards=0)


def test_configure_determinism_sets_seed_and_backend_flags(monkeypatch):
    seeds = []
    deterministic_algorithms = []
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(common.torch, "manual_seed", lambda seed: seeds.append(("cpu", seed)))
    monkeypatch.setattr(
        common.torch.cuda,
        "manual_seed_all",
        lambda seed: seeds.append(("cuda", seed)),
    )
    monkeypatch.setattr(
        common.torch,
        "use_deterministic_algorithms",
        deterministic_algorithms.append,
    )
    monkeypatch.setattr(common.torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(common.torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(common.torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(common.torch.backends.cudnn, "allow_tf32", True)

    configure_determinism(seed=37)

    assert seeds == [("cpu", 37), ("cuda", 37)]
    assert deterministic_algorithms == [True]
    assert common.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert common.torch.backends.cudnn.benchmark is False
    assert common.torch.backends.cudnn.deterministic is True
    assert common.torch.backends.cuda.matmul.allow_tf32 is False
    assert common.torch.backends.cudnn.allow_tf32 is False


@pytest.mark.parametrize("requested", ["cuda", "cuda:0"])
def test_require_cuda_returns_canonical_device_zero(monkeypatch, requested):
    allocations = []
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        common.torch,
        "empty",
        lambda *shape, device: allocations.append((shape, device)),
    )

    resolved = require_cuda(requested)

    assert resolved == torch.device("cuda:0")
    assert allocations == [((1,), torch.device("cuda:0"))]


@pytest.mark.parametrize("requested", ["cpu", "cuda:1", "cuda:2"])
def test_require_cuda_rejects_every_device_other_than_cuda_zero(monkeypatch, requested):
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        common.torch,
        "empty",
        lambda *args, **kwargs: pytest.fail("Rejected devices must not be allocated"),
    )
    with pytest.raises(ValueError, match="CUDA device 0"):
        require_cuda(requested)


def test_require_cuda_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        common.torch,
        "empty",
        lambda *args, **kwargs: pytest.fail("CUDA allocation must not be attempted"),
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        require_cuda("cuda:0")


class _FakeDistribution:
    def __init__(self, version, direct_url=None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename):
        assert filename == "direct_url.json"
        return self._direct_url


def _mock_capture_runtime(monkeypatch):
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="stok-commit\n"),
    )
    monkeypatch.setattr(common.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(common.torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(common.torch.version, "cuda", "12.8")
    monkeypatch.setattr(common.torch.cuda, "get_device_name", lambda device: "RTX A6000")
    monkeypatch.setattr(common.torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(common.torch.backends.cudnn, "version", lambda: 91002)
    monkeypatch.setattr(
        common,
        "_nvidia_driver_version",
        lambda index: "575.57.08",
        raising=False,
    )
    monkeypatch.setattr(common, "_cuda_runtime_version", lambda: "12.8", raising=False)


def test_capture_environment_records_cuda_package_and_upstream_provenance(monkeypatch):
    direct_url = json.dumps(
        {
            "subdirectory": "gcp-vqvae",
            "url": "https://github.com/mahdip72/vq_encoder_decoder.git",
            "vcs_info": {
                "commit_id": "upstream-commit",
                "requested_revision": "master",
                "vcs": "git",
            },
        }
    )
    distributions = {
        "gcp-vqvae": _FakeDistribution("0.2.2", direct_url),
        "graphein": _FakeDistribution("1.7.7"),
    }

    _mock_capture_runtime(monkeypatch)
    monkeypatch.setattr(
        common.importlib.metadata,
        "distribution",
        lambda name: (
            distributions[name]
            if name in distributions
            else (_ for _ in ()).throw(common.importlib.metadata.PackageNotFoundError(name))
        ),
    )
    origins = {
        "gcp_vqvae": "/env/site-packages/gcp_vqvae/__init__.py",
        "graphein": "/env/site-packages/graphein/__init__.py",
    }
    monkeypatch.setattr(
        common.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=origins[name]) if name in origins else None,
    )

    environment = capture_environment(torch.device("cuda:0"))

    assert environment["git_commit"] == "stok-commit"
    assert environment["platform"] == "test-platform"
    assert environment["cuda"] == {
        "device": {"index": 0, "name": "RTX A6000", "capability": [8, 6]},
        "versions": {
            "nvidia_driver": "575.57.08",
            "cuda_runtime": "12.8",
            "torch_build_cuda": "12.8",
            "cudnn": 91002,
        },
    }
    assert environment["packages"]["gcp-vqvae"] == "0.2.2"
    assert environment["package_provenance"]["gcp-vqvae"] == {
        "metadata_present": True,
        "importable": True,
        "module": "gcp_vqvae",
        "origin": "/env/site-packages/gcp_vqvae/__init__.py",
        "version": "0.2.2",
        "source": {
            "url": "https://github.com/mahdip72/vq_encoder_decoder.git",
            "subdirectory": "gcp-vqvae",
            "vcs": "git",
            "requested_revision": "master",
            "commit_id": "upstream-commit",
        },
    }
    assert environment["upstream_vq_encoder_decoder"] == {
        "repository": "vq_encoder_decoder",
        "distribution": "gcp-vqvae",
        "distribution_version": "0.2.2",
        "module": "gcp_vqvae",
        "module_origin": "/env/site-packages/gcp_vqvae/__init__.py",
        "source_url": "https://github.com/mahdip72/vq_encoder_decoder.git",
        "source_revision": "upstream-commit",
        "requested_revision": "master",
        "subdirectory": "gcp-vqvae",
    }


def test_capture_environment_records_missing_packages_deterministically(monkeypatch):
    _mock_capture_runtime(monkeypatch)
    monkeypatch.setattr(
        common.importlib.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(common.importlib.metadata.PackageNotFoundError(name)),
    )
    monkeypatch.setattr(common.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(common, "_nvidia_driver_version", lambda index: None)
    monkeypatch.setattr(common, "_cuda_runtime_version", lambda: None)

    environment = capture_environment(torch.device("cuda:0"))

    assert environment["packages"]["torch-cluster"] is None
    assert environment["package_provenance"]["torch-cluster"] == {
        "metadata_present": False,
        "importable": False,
        "module": "torch_cluster",
        "origin": None,
        "version": None,
        "source": {
            "url": None,
            "subdirectory": None,
            "vcs": None,
            "requested_revision": None,
            "commit_id": None,
        },
    }
    assert environment["cuda"]["versions"]["nvidia_driver"] is None
    assert environment["cuda"]["versions"]["cuda_runtime"] is None
    assert environment["upstream_vq_encoder_decoder"]["source_revision"] is None


def test_cuda_runtime_version_reads_loaded_cuda_runtime(monkeypatch):
    class FakeCudaRuntime:
        @staticmethod
        def cudaRuntimeGetVersion(pointer):
            pointer._obj.value = 12080
            return 0

    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libcudart-test.so")
    monkeypatch.setattr(ctypes, "CDLL", lambda name: FakeCudaRuntime())
    monkeypatch.setattr(
        common.torch.cuda,
        "cudart",
        lambda: pytest.fail("Runtime version must come from the CUDA runtime library"),
    )

    assert common._cuda_runtime_version() == "12.8"


def test_cuda_runtime_version_is_none_when_runtime_is_absent(monkeypatch):
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: (_ for _ in ()).throw(OSError(name)))
    monkeypatch.setattr(
        common.torch.cuda,
        "cudart",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")),
    )

    assert common._cuda_runtime_version() is None


def test_nvidia_driver_version_handles_available_and_absent_smi(monkeypatch):
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="575.57.08\n"),
    )
    assert common._nvidia_driver_version(0) == "575.57.08"

    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(FileNotFoundError(command[0])),
    )
    assert common._nvidia_driver_version(0) is None


def test_atomic_write_json_replaces_target(tmp_path):
    target = tmp_path / "nested" / "result.json"
    atomic_write_json(target, {"status": "first"})
    atomic_write_json(target, {"status": "second"})
    assert json.loads(target.read_text()) == {"status": "second"}
    assert not target.with_suffix(".json.tmp").exists()


def test_preprocessing_audit_preserves_all_reference_samples(tmp_path):
    path = tmp_path / "complex.cif"
    path.write_text("structure")
    inputs = discover_inputs(tmp_path)
    reference_samples = [
        {
            "source_path": str(path),
            "pid": "0_complex_chain_id_A",
            "seq": "AC",
            "coords": [[[0.0] * 3] * 4] * 2,
        },
        {
            "source_path": str(path),
            "pid": "0_complex_chain_id_B",
            "seq": "GGG",
            "coords": [[[0.0] * 3] * 4] * 3,
        },
    ]

    records = build_preprocessing_audit(
        inputs,
        reference_samples,
        stok_parser=lambda _: SimpleNamespace(protein_sequence="AC", chain_id="A"),
    )

    assert len(records) == 1
    assert [sample.pid for sample in records[0].reference_samples] == [
        "0_complex_chain_id_A",
        "0_complex_chain_id_B",
    ]
    assert records[0].stok_sequence == "AC"
    assert records[0].stok_chain_id == "A"


def test_preprocessing_audit_records_parser_exception(tmp_path):
    path = tmp_path / "bad.cif"
    path.write_text("structure")

    def fail(_):
        raise ImportError("three_to_one")

    record = build_preprocessing_audit(discover_inputs(tmp_path), [], stok_parser=fail)[0]
    assert record.stok_error_type == "ImportError"
    assert record.stok_error_message == "three_to_one"
    assert record.stok_sequence is None
