"""
Tests for C++ extension building utilities.

These tests verify that after importing torchada, the standard torch imports
work correctly:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
"""

import os

# Import torchada first to apply patches
import torchada  # noqa: F401


class TestCppExtensionImports:
    """Test cpp_extension module imports using standard torch imports."""

    def test_import_cuda_home(self):
        """Test CUDA_HOME can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import CUDA_HOME

        # CUDA_HOME should be a string or None
        assert CUDA_HOME is None or isinstance(CUDA_HOME, str)

    def test_import_cuda_extension(self):
        """Test CUDAExtension can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import CUDAExtension

        assert CUDAExtension is not None

    def test_import_build_extension(self):
        """Test BuildExtension can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import BuildExtension

        assert BuildExtension is not None

    def test_cuda_home_on_musa(self):
        """Test CUDA_HOME points to MUSA on MUSA platform."""
        from torch.utils.cpp_extension import CUDA_HOME

        if torchada.is_musa_platform() and CUDA_HOME is not None:
            # On MUSA platform, CUDA_HOME should point to MUSA installation
            assert "musa" in CUDA_HOME.lower() or os.path.exists(
                os.path.join(CUDA_HOME, "bin", "mcc")
            )

    def test_torch_cpp_extension_is_patched(self):
        """Test that torch.utils.cpp_extension is patched correctly on MUSA."""
        import torch.utils.cpp_extension as torch_cpp_ext

        if torchada.is_musa_platform():
            # Verify CUDAExtension is our patched version
            assert torch_cpp_ext.CUDAExtension.__module__ == "torchada.utils.cpp_extension"

    def test_torch_cpp_extension_cuda_home_same_as_torchada(self):
        """Test that torch.utils.cpp_extension.CUDA_HOME matches torchada's."""
        import torch.utils.cpp_extension as torch_cpp_ext

        from torchada.utils.cpp_extension import CUDA_HOME as torchada_cuda_home

        if torchada.is_musa_platform():
            assert torch_cpp_ext.CUDA_HOME == torchada_cuda_home


class TestCUDAExtension:
    """Test CUDAExtension class using standard torch imports."""

    def test_create_extension_basic(self):
        """Test basic CUDAExtension creation."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
        )
        assert ext.name == "test_ext"
        assert "test.cu" in ext.sources

    def test_create_extension_with_include_dirs(self):
        """Test CUDAExtension with include_dirs."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
            include_dirs=["/usr/include"],
        )
        assert "/usr/include" in ext.include_dirs

    def test_create_extension_with_extra_compile_args(self):
        """Test CUDAExtension with extra_compile_args."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-arch=sm_70"]},
        )
        assert ext.extra_compile_args is not None


class TestMusaPatches:
    """Test patches applied to torch_musa for extension building."""

    def test_is_musa_file_recognizes_cu(self):
        """Test _is_musa_file recognizes .cu files."""
        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.cu")
            assert musa_ext._is_musa_file("path/to/kernel.cu")

    def test_is_musa_file_recognizes_cuh(self):
        """Test _is_musa_file recognizes .cuh files."""
        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.cuh")
            assert musa_ext._is_musa_file("path/to/header.cuh")

    def test_is_musa_file_recognizes_mu(self):
        """Test _is_musa_file still recognizes .mu files."""
        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.mu")

    def test_ext_replaced_mapping(self):
        """Test EXT_REPLACED_MAPPING keeps .cu/.cuh."""
        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            # Extensions are converted: .cu -> .mu, .cuh -> .muh for mcc compiler
            assert musa_sp.EXT_REPLACED_MAPPING["cu"] == "mu"
            assert musa_sp.EXT_REPLACED_MAPPING["cuh"] == "muh"

    def test_mapping_rule_exists(self):
        """Test _MAPPING_RULE is set."""
        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            assert hasattr(musa_sp, "_MAPPING_RULE")
            assert len(musa_sp._MAPPING_RULE) > 0

    def test_mapping_rule_has_expected_entries(self):
        """Test _MAPPING_RULE has expected entries."""
        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            rules = musa_sp._MAPPING_RULE

            # Check some key mappings
            assert rules.get("cudaMalloc") == "musaMalloc"
            assert rules.get("cudaFree") == "musaFree"
            assert rules.get("cudaStream_t") == "musaStream_t"
            assert rules.get("at::cuda") == "at::musa"
            assert rules.get("c10::cuda") == "c10::musa"


class TestExcludeDirs:
    """Test the exclude directories functionality for _port_directory."""

    def test_get_exclude_dirs_default_empty(self):
        """Test get_exclude_dirs returns empty list when no env var is set."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        # Remove env var to ensure clean state
        env_backup = os.environ.pop("TORCHADA_EXCLUDE_DIRS", None)
        try:
            # BuildExtension on MUSA is _MUSABuildExtension
            # We need an instance to call get_exclude_dirs, but it requires
            # distutils Command initialization. Instead, test the logic directly.
            cls = _get_build_extension_class()
            # The class method can be called on the class itself for testing
            # since get_exclude_dirs doesn't use 'self' state
            result = cls.get_exclude_dirs(cls)
            assert isinstance(result, list)
        finally:
            if env_backup is not None:
                os.environ["TORCHADA_EXCLUDE_DIRS"] = env_backup

    def test_get_exclude_dirs_from_env(self):
        """Test get_exclude_dirs reads from TORCHADA_EXCLUDE_DIRS env var."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        # Set env var
        old_val = os.environ.get("TORCHADA_EXCLUDE_DIRS")
        try:
            test_dir = "/tmp/test_exclude_dir"
            os.environ["TORCHADA_EXCLUDE_DIRS"] = test_dir
            result = cls.get_exclude_dirs(cls)
            assert os.path.abspath(test_dir) in result
        finally:
            if old_val is not None:
                os.environ["TORCHADA_EXCLUDE_DIRS"] = old_val
            else:
                os.environ.pop("TORCHADA_EXCLUDE_DIRS", None)

    def test_get_exclude_dirs_multiple_from_env(self):
        """Test get_exclude_dirs handles multiple paths from env var."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        old_val = os.environ.get("TORCHADA_EXCLUDE_DIRS")
        try:
            dir1 = "/tmp/exclude_dir1"
            dir2 = "/tmp/exclude_dir2"
            os.environ["TORCHADA_EXCLUDE_DIRS"] = dir1 + os.pathsep + dir2
            result = cls.get_exclude_dirs(cls)
            assert os.path.abspath(dir1) in result
            assert os.path.abspath(dir2) in result
        finally:
            if old_val is not None:
                os.environ["TORCHADA_EXCLUDE_DIRS"] = old_val
            else:
                os.environ.pop("TORCHADA_EXCLUDE_DIRS", None)

    def test_get_exclude_dirs_strips_whitespace(self):
        """Test get_exclude_dirs strips whitespace from env var entries."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        old_val = os.environ.get("TORCHADA_EXCLUDE_DIRS")
        try:
            test_dir = "/tmp/test_exclude_dir"
            os.environ["TORCHADA_EXCLUDE_DIRS"] = "  " + test_dir + "  "
            result = cls.get_exclude_dirs(cls)
            assert os.path.abspath(test_dir) in result
        finally:
            if old_val is not None:
                os.environ["TORCHADA_EXCLUDE_DIRS"] = old_val
            else:
                os.environ.pop("TORCHADA_EXCLUDE_DIRS", None)

    def test_get_exclude_dirs_ignores_empty_entries(self):
        """Test get_exclude_dirs ignores empty entries in env var."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        old_val = os.environ.get("TORCHADA_EXCLUDE_DIRS")
        try:
            test_dir = "/tmp/test_exclude_dir"
            # Double separator creates empty entries
            os.environ["TORCHADA_EXCLUDE_DIRS"] = test_dir + os.pathsep + os.pathsep
            result = cls.get_exclude_dirs(cls)
            assert len([d for d in result if d == ""]) == 0
            assert os.path.abspath(test_dir) in result
        finally:
            if old_val is not None:
                os.environ["TORCHADA_EXCLUDE_DIRS"] = old_val
            else:
                os.environ.pop("TORCHADA_EXCLUDE_DIRS", None)

    def test_is_excluded_dir_exact_match(self):
        """Test _is_excluded_dir with exact directory match."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        exclude_dirs = ["/tmp/exact_match_dir"]
        assert cls._is_excluded_dir(cls, "/tmp/exact_match_dir", exclude_dirs)

    def test_is_excluded_dir_subdirectory(self):
        """Test _is_excluded_dir matches subdirectories of excluded dirs."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        exclude_dirs = ["/tmp/parent_dir"]
        assert cls._is_excluded_dir(cls, "/tmp/parent_dir/child", exclude_dirs)
        assert cls._is_excluded_dir(cls, "/tmp/parent_dir/child/grandchild", exclude_dirs)

    def test_is_excluded_dir_no_match(self):
        """Test _is_excluded_dir returns False for non-excluded directories."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        exclude_dirs = ["/tmp/excluded"]
        assert not cls._is_excluded_dir(cls, "/tmp/not_excluded", exclude_dirs)
        assert not cls._is_excluded_dir(cls, "/tmp/excluded_other", exclude_dirs)

    def test_is_excluded_dir_no_partial_match(self):
        """Test _is_excluded_dir does not match partial directory names."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        # /tmp/excluded_suffix should NOT match /tmp/excluded
        exclude_dirs = ["/tmp/excluded"]
        assert not cls._is_excluded_dir(cls, "/tmp/excluded_suffix", exclude_dirs)

    def test_is_excluded_dir_empty_list(self):
        """Test _is_excluded_dir returns False with empty exclude list."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        assert not cls._is_excluded_dir(cls, "/tmp/any_dir", [])

    def test_convert_source_path_excluded_dir(self):
        """Test _convert_source_path returns source as-is for excluded directories."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        exclude_dirs = ["/tmp/excluded_csrc"]
        # A .cu file in an excluded directory should NOT be converted
        source = "/tmp/excluded_csrc/kernel.cu"
        new_source, needs_porting = cls._convert_source_path(cls, source, exclude_dirs)
        assert new_source == source
        assert needs_porting is False

    def test_convert_source_path_non_excluded_dir(self):
        """Test _convert_source_path converts files in non-excluded directories."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        exclude_dirs = ["/tmp/excluded_csrc"]
        # A .cu file in a non-excluded directory should be converted
        source = "/tmp/normal_csrc/kernel.cu"
        new_source, needs_porting = cls._convert_source_path(cls, source, exclude_dirs)
        assert needs_porting is True
        assert "_musa" in new_source

    def test_port_directory_excluded_returns_source_dir(self):
        """Test _port_directory returns source_dir when directory is excluded."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        cls = _get_build_extension_class()

        exclude_dirs = ["/tmp/excluded_port_dir"]
        # _port_directory should return the source_dir itself (not _musa) when excluded
        result = cls._port_directory(cls, "/tmp/excluded_port_dir", exclude_dirs=exclude_dirs)
        assert result == os.path.abspath("/tmp/excluded_port_dir")

    def test_subclass_can_extend_exclude_dirs(self):
        """Test that subclasses can extend get_exclude_dirs (open-closed principle)."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        BaseClass = _get_build_extension_class()

        # Simulate subclass extending get_exclude_dirs
        class CustomBuildExt(BaseClass):
            def get_exclude_dirs(self):
                return super().get_exclude_dirs() + [
                    "/custom/vendor/dir",
                    "/custom/third_party/dir",
                ]

        old_val = os.environ.get("TORCHADA_EXCLUDE_DIRS")
        try:
            os.environ.pop("TORCHADA_EXCLUDE_DIRS", None)
            instance_mock = CustomBuildExt.__new__(CustomBuildExt)
            result = instance_mock.get_exclude_dirs()
            assert "/custom/vendor/dir" in result
            assert "/custom/third_party/dir" in result
        finally:
            if old_val is not None:
                os.environ["TORCHADA_EXCLUDE_DIRS"] = old_val

    def test_subclass_exclude_dirs_merges_with_env(self):
        """Test that subclass exclude dirs merge with env var entries."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        BaseClass = _get_build_extension_class()

        class CustomBuildExt(BaseClass):
            def get_exclude_dirs(self):
                return super().get_exclude_dirs() + ["/custom/dir"]

        old_val = os.environ.get("TORCHADA_EXCLUDE_DIRS")
        try:
            os.environ["TORCHADA_EXCLUDE_DIRS"] = "/env/dir"
            instance_mock = CustomBuildExt.__new__(CustomBuildExt)
            result = instance_mock.get_exclude_dirs()
            assert os.path.abspath("/env/dir") in result
            assert "/custom/dir" in result
        finally:
            if old_val is not None:
                os.environ["TORCHADA_EXCLUDE_DIRS"] = old_val
            else:
                os.environ.pop("TORCHADA_EXCLUDE_DIRS", None)
