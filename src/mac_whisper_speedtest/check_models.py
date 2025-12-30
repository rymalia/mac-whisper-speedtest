"""Model checker and downloader for Whisper implementations.

This module provides functionality to check if models are cached locally,
verify their integrity, and download missing models.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import shutil

import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt, Confirm

from mac_whisper_speedtest.implementations.base import ModelInfo

log = structlog.get_logger(__name__)
console = Console()


@dataclass
class ModelStatus:
    """Status of a model for a specific implementation."""
    implementation: str
    model_info: ModelInfo
    hf_cache_status: str  # "complete", "missing", "incomplete", "n/a"
    local_cache_status: str  # "complete", "missing", "incomplete", "n/a"
    hf_cache_size_mb: Optional[float] = None
    local_cache_size_mb: Optional[float] = None


class ModelChecker:
    """Check and manage model cache status across implementations."""

    def __init__(self, verify_method: Optional[str] = None, verbose: bool = False):
        """Initialize model checker.

        Args:
            verify_method: Force specific verification method ('cache-check', 'timeout', or None for auto)
            verbose: Show timing information during verification
        """
        self.hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # Cache verification results to avoid loading same model multiple times
        self._verification_cache = {}
        self.verify_method_override = verify_method
        self.verbose = verbose

    def _calculate_timeout(self, model_name: str, model_info: ModelInfo) -> int:
        """Calculate appropriate timeout based on model size.

        Uses model-size-aware timeout scaling:
        - tiny, base, small, medium: 15 seconds
        - large models: 30 seconds

        Args:
            model_name: Standard model name (tiny, small, medium, large)
            model_info: ModelInfo from implementation (may specify custom timeout)

        Returns:
            Timeout in seconds
        """
        # If implementation specifies timeout, use it
        if model_info.timeout_seconds is not None:
            return model_info.timeout_seconds

        # Otherwise calculate based on model name
        model_lower = model_name.lower()
        if "large" in model_lower:
            return 30
        else:
            return 15

    def _verify_with_timeout(self, impl_instance, model_name: str, timeout_seconds: int) -> tuple[str, Optional[float]]:
        """Verify model by loading with timeout protection.

        This method attempts to load the model with a timeout. If the load completes
        within the timeout, the model is verified as complete. If it times out, the
        model is likely missing and would trigger a download.

        Args:
            impl_instance: The implementation instance
            model_name: The model size name (e.g., "small", "large")
            timeout_seconds: Maximum time to wait for model load

        Returns:
            Tuple of (status, verification_time_seconds)
            Status: "complete", "incomplete", or "missing"
        """
        import time
        import signal
        from contextlib import contextmanager

        cache_key = (impl_instance.__class__.__name__, model_name)
        if cache_key in self._verification_cache:
            return self._verification_cache[cache_key], None

        @contextmanager
        def timeout_handler(seconds):
            """Context manager for timeout using SIGALRM."""
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Verification timed out after {seconds}s")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        start_time = time.time() if self.verbose else None

        try:
            with timeout_handler(timeout_seconds):
                # Try to load the model using the implementation's own load_model method
                impl_instance.load_model(model_name)

                # If load succeeded, immediately cleanup to release memory
                if hasattr(impl_instance, 'cleanup'):
                    impl_instance.cleanup()

                elapsed = time.time() - start_time if start_time else None

                # Model loaded successfully - it's complete
                result = "complete"
                self._verification_cache[cache_key] = result

                if self.verbose and elapsed:
                    log.info(f"Verified {impl_instance.__class__.__name__} {model_name} in {elapsed:.2f}s")

                return result, elapsed

        except TimeoutError as e:
            # Timeout indicates model not cached (would trigger download)
            elapsed = time.time() - start_time if start_time else None
            log.warning(f"Timeout verifying {impl_instance.__class__.__name__} {model_name}: {e}")
            result = "incomplete"
            self._verification_cache[cache_key] = result

            if self.verbose and elapsed:
                log.info(f"Verification timed out for {impl_instance.__class__.__name__} {model_name} after {elapsed:.2f}s")

            return result, elapsed

        except FileNotFoundError:
            # Model files are missing
            elapsed = time.time() - start_time if start_time else None
            result = "incomplete"
            self._verification_cache[cache_key] = result
            return result, elapsed

        except Exception as e:
            # Other errors could indicate corruption or incomplete download
            elapsed = time.time() - start_time if start_time else None
            error_msg = str(e).lower()

            if any(indicator in error_msg for indicator in
                   ["incomplete", "corrupted", "missing", "not found", "no such file"]):
                result = "incomplete"
            else:
                log.warning(f"Error loading {impl_instance.__class__.__name__} {model_name}: {e}")
                result = "incomplete"

            self._verification_cache[cache_key] = result
            return result, elapsed

    def check_all_models(self, model_size: str, implementations: List) -> List[ModelStatus]:
        """Check status of all models for the given size.

        Args:
            model_size: The model size to check (tiny, small, medium, large)
            implementations: List of implementation classes to check

        Returns:
            List of ModelStatus for each implementation
        """
        statuses = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Checking {model_size} models...", total=len(implementations))

            for impl_class in implementations:
                try:
                    impl = impl_class()
                    model_info = impl.get_model_info(model_size)

                    # Check HuggingFace cache
                    hf_status, hf_size = self._check_hf_cache(model_info, impl, model_size)

                    # Check local cache
                    local_status, local_size = self._check_local_cache(model_info, impl, model_size)

                    status = ModelStatus(
                        implementation=impl_class.__name__,
                        model_info=model_info,
                        hf_cache_status=hf_status,
                        local_cache_status=local_status,
                        hf_cache_size_mb=hf_size,
                        local_cache_size_mb=local_size,
                    )
                    statuses.append(status)

                except Exception as e:
                    log.error(f"Error checking {impl_class.__name__}: {e}")

                progress.update(task, advance=1)

        return statuses

    def _check_hf_cache(self, model_info: ModelInfo, impl_instance, model_name: str) -> tuple[str, Optional[float]]:
        """Check if model exists in default HuggingFace cache.

        Args:
            model_info: ModelInfo from the implementation
            impl_instance: The implementation instance
            model_name: The model size name (e.g., "small", "large")

        Returns:
            Tuple of (status, size_in_mb)
            Status: "complete", "missing", "incomplete", or "n/a"
        """
        if not model_info.repo_id:
            return "n/a", None

        if model_info.verification_method == "huggingface":
            # Always check default HF cache for this column
            return self._verify_hf_model(model_info.repo_id, cache_dir=None,
                                        impl_instance=impl_instance, model_name=model_name)

        return "n/a", None

    def _verify_hf_model(self, repo_id: str, cache_dir: Optional[str] = None,
                        impl_instance=None, model_name: str = None) -> tuple[str, Optional[float]]:
        """Verify HuggingFace model using huggingface_hub tools.

        Args:
            repo_id: The HuggingFace repository ID to check
            cache_dir: Specific cache directory to check, or None for default HF cache
            impl_instance: The implementation instance (for load-based verification)
            model_name: The model size name (for load-based verification)

        Returns:
            Tuple of (status, size_in_mb)
        """
        try:
            from huggingface_hub import scan_cache_dir

            # Check specified cache directory or default HF cache
            if cache_dir:
                cache_info = scan_cache_dir(cache_dir=cache_dir)
            else:
                # Check default HF cache (~/.cache/huggingface/hub/)
                cache_info = scan_cache_dir()

            # Find the repo in cache
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    # Calculate total size
                    total_size = repo.size_on_disk
                    size_mb = total_size / (1024 * 1024)

                    # Quick check for .incomplete markers first (fast path)
                    repo_path = repo.repo_path
                    blobs_dir = repo_path / "blobs"
                    if blobs_dir.exists():
                        incomplete_files = list(blobs_dir.glob("*.incomplete"))
                        if incomplete_files:
                            log.warning(f"Incomplete HF model {repo_id}: Found .incomplete markers")
                            return "incomplete", size_mb

                    # Verify by actually loading the model (uses exact same code as benchmark)
                    if impl_instance and model_name:
                        is_complete = self._verify_by_loading(impl_instance, model_name)
                        if is_complete == "complete":
                            return "complete", size_mb
                        else:
                            return is_complete, size_mb

                    # Fallback: if no implementation provided, assume complete
                    return "complete", size_mb

            return "missing", None

        except ImportError:
            log.warning("huggingface_hub not available for cache verification")
            return "n/a", None
        except Exception as e:
            log.error(f"Error verifying HF model {repo_id}: {e}")
            return "incomplete", None

    def _verify_by_loading(self, impl_instance, model_name: str) -> str:
        """Verify model completeness by actually loading it.

        This uses timeout-protected loading to prevent triggering long downloads
        during verification. Uses the exact same code path as benchmark.

        Args:
            impl_instance: The implementation instance
            model_name: The model size name (e.g., "small", "large")

        Returns:
            Status: "complete", "incomplete", or "error"
        """
        # Get model info to calculate timeout
        model_info = impl_instance.get_model_info(model_name)
        timeout = self._calculate_timeout(model_name, model_info)

        # Check if user forced a specific verification method
        if self.verify_method_override == "timeout":
            # Force timeout-based verification
            status, _ = self._verify_with_timeout(impl_instance, model_name, timeout)
            return status
        elif self.verify_method_override == "cache-check":
            # Force cache-check only (no load attempt)
            # This is for debugging - just check for .incomplete markers
            log.info(f"Using cache-check only for {impl_instance.__class__.__name__} (no load attempt)")
            return "incomplete"  # Conservative: assume incomplete without loading
        else:
            # Auto mode: use timeout-protected verification for HF implementations
            # This prevents downloads during verification
            if model_info.verification_method == "huggingface":
                status, _ = self._verify_with_timeout(impl_instance, model_name, timeout)
                return status
            else:
                # For non-HF implementations, use timeout as well for consistency
                status, _ = self._verify_with_timeout(impl_instance, model_name, timeout)
                return status

    def _check_local_cache(self, model_info: ModelInfo, impl_instance, model_name: str) -> tuple[str, Optional[float]]:
        """Check if model exists in local cache paths or custom HF cache directory.

        Args:
            model_info: ModelInfo from the implementation
            impl_instance: The implementation instance
            model_name: The model size name (e.g., "small", "large")

        Returns:
            Tuple of (status, size_in_mb)
        """
        # For HF-based implementations that use a custom cache directory (MLX, FasterWhisper)
        # Check that directory instead of cache_paths
        if model_info.hf_cache_dir and model_info.repo_id:
            return self._verify_hf_model(model_info.repo_id, cache_dir=model_info.hf_cache_dir,
                                        impl_instance=impl_instance, model_name=model_name)

        # For other implementations, check cache_paths
        if not model_info.cache_paths:
            return "n/a", None

        # Check if all cache paths exist
        all_exist = all(path.exists() for path in model_info.cache_paths)

        if not all_exist:
            # Check if any exist (partial)
            any_exist = any(path.exists() for path in model_info.cache_paths)
            if any_exist:
                return "incomplete", self._calculate_total_size(model_info.cache_paths)
            return "missing", None

        # All paths exist, calculate size
        total_size_mb = self._calculate_total_size(model_info.cache_paths)

        # For local path-based models, verify by loading
        # This ensures we use the same verification as benchmark
        is_complete = self._verify_by_loading(impl_instance, model_name)
        if is_complete != "complete":
            return is_complete, total_size_mb

        # Verify size if expected size is provided (secondary check)
        if model_info.expected_size_mb and model_info.verification_method == "size":
            # Allow 10% variance
            expected = model_info.expected_size_mb
            if total_size_mb < expected * 0.9:
                return "incomplete", total_size_mb

        return "complete", total_size_mb

    def _calculate_total_size(self, paths: List[Path]) -> float:
        """Calculate total size of all paths in MB."""
        total_bytes = 0

        for path in paths:
            if not path.exists():
                continue

            if path.is_file():
                total_bytes += path.stat().st_size
            elif path.is_dir():
                # Recursively calculate directory size
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        try:
                            total_bytes += file_path.stat().st_size
                        except OSError:
                            pass

        return total_bytes / (1024 * 1024)

    def print_status_table(self, statuses: List[ModelStatus], model_size: str):
        """Print a rich table showing model status."""
        table = Table(title=f"Model Status for '{model_size}' Models")

        table.add_column("Implementation", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("HF Hub Cache", style="magenta")
        table.add_column("Local Cache", style="green")
        table.add_column("Disk Usage (MB)", justify="right")

        for status in statuses:
            # Format status with emojis
            hf_icon = self._status_icon(status.hf_cache_status)
            local_icon = self._status_icon(status.local_cache_status)

            # Use local size if available, otherwise HF size
            size_str = ""
            if status.local_cache_size_mb:
                size_str = f"{status.local_cache_size_mb:.1f}"
            elif status.hf_cache_size_mb:
                size_str = f"{status.hf_cache_size_mb:.1f}"

            # Shorten implementation name
            impl_name = status.implementation.replace("Implementation", "")

            table.add_row(
                impl_name,
                status.model_info.model_name,
                f"{hf_icon} {status.hf_cache_status}",
                f"{local_icon} {status.local_cache_status}",
                size_str
            )

        console.print(table)

    def _status_icon(self, status: str) -> str:
        """Get emoji icon for status."""
        icons = {
            "complete": "✓",
            "missing": "✗",
            "incomplete": "⚠",
            "n/a": "—"
        }
        return icons.get(status, "?")

    def print_summary(self, statuses: List[ModelStatus]):
        """Print summary statistics."""
        # A model is ready if it's complete in either HF cache or local cache
        ready = sum(1 for s in statuses if
            s.local_cache_status == "complete" or s.hf_cache_status == "complete")

        # A model is missing if both caches are missing/n/a (and neither is complete)
        missing = sum(1 for s in statuses if
            s.local_cache_status in ["missing", "n/a"] and
            s.hf_cache_status in ["missing", "n/a"])

        # A model is incomplete if either cache is incomplete and neither is complete
        incomplete = sum(1 for s in statuses if
            (s.local_cache_status == "incomplete" or s.hf_cache_status == "incomplete") and
            s.local_cache_status != "complete" and s.hf_cache_status != "complete")

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  • {ready} models ready")
        console.print(f"  • {missing} models missing")
        console.print(f"  • {incomplete} models incomplete")

        # Calculate total download size needed
        total_needed_mb = 0
        for status in statuses:
            if status.local_cache_status in ["missing", "incomplete"]:
                if status.hf_cache_size_mb:
                    total_needed_mb += status.hf_cache_size_mb
                elif status.model_info.expected_size_mb:
                    total_needed_mb += status.model_info.expected_size_mb

        if total_needed_mb > 0:
            total_gb = total_needed_mb / 1024
            console.print(f"\n[yellow]Total download size needed: {total_gb:.2f} GB[/yellow]")

            # Check available disk space
            stat = shutil.disk_usage(Path.home())
            free_gb = stat.free / (1024**3)
            console.print(f"[green]Available disk space: {free_gb:.1f} GB[/green]")

    async def download_missing_models(
        self,
        statuses: List[ModelStatus],
        model_size: str,
        implementations: List,
        implementations_to_download: Optional[Set[str]] = None
    ):
        """Download missing models.

        Args:
            statuses: List of model statuses
            model_size: The model size being checked
            implementations: List of implementation classes
            implementations_to_download: Optional set of implementation names to download.
                                        If None, downloads all missing models.
        """
        to_download = [
            s for s in statuses
            if (s.local_cache_status in ["missing", "incomplete"] or s.hf_cache_status in ["missing", "incomplete"])
            and (implementations_to_download is None or s.implementation in implementations_to_download)
        ]

        if not to_download:
            console.print("[green]All selected models are already downloaded![/green]")
            return

        console.print(f"\n[bold]Downloading {len(to_download)} models...[/bold]")

        # TODO: Implement parallel download
        # For now, download sequentially
        for status in to_download:
            await self._download_model(status)

    async def _download_model(self, status: ModelStatus):
        """Download a single model."""
        console.print(f"\n[cyan]Downloading {status.implementation}...[/cyan]")

        if status.model_info.download_trigger == "bridge":
            # Use dummy audio to trigger bridge download
            await self._trigger_bridge_download(status)
        elif status.model_info.download_trigger == "auto":
            # HuggingFace auto-download
            await self._trigger_hf_download(status)
        elif status.model_info.download_trigger == "native":
            # Use implementation's native download (e.g., whisper-mps)
            await self._trigger_native_download(status)
        else:
            console.print(f"[yellow]Manual download required for {status.implementation}[/yellow]")

    async def _trigger_bridge_download(self, status: ModelStatus):
        """Trigger download by running bridge with dummy audio."""
        import tempfile
        import subprocess
        import numpy as np
        import soundfile as sf
        from pathlib import Path

        console.print(f"  Triggering {status.implementation} download via bridge...")

        # Determine bridge path based on implementation
        project_root = Path(__file__).parent.parent.parent
        if "WhisperKit" in status.implementation:
            bridge_path = project_root / "tools" / "whisperkit-bridge" / ".build" / "release" / "whisperkit-bridge"
            model_arg = status.model_info.model_name.split()[0]  # Extract model name from "small + CoreML"
        elif "FluidAudio" in status.implementation:
            bridge_path = project_root / "tools" / "fluidaudio-bridge" / ".build" / "release" / "fluidaudio-bridge"
            model_arg = None  # FluidAudio doesn't need model arg
        else:
            console.print(f"  [yellow]Unknown bridge type for {status.implementation}[/yellow]")
            return

        if not bridge_path.exists():
            console.print(f"  [red]Bridge not found at {bridge_path}[/red]")
            console.print(f"  [yellow]Please build it first: cd {bridge_path.parent.parent} && swift build -c release[/yellow]")
            return

        try:
            # Create dummy audio (1 second of silence at 16kHz)
            sample_rate = 16000
            duration = 1.0
            dummy_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, dummy_audio, sample_rate)

            try:
                # Build command
                cmd = [str(bridge_path), temp_path, "--format", "json"]
                if model_arg:
                    cmd.extend(["--model", model_arg])

                console.print(f"  Running bridge to download model...")

                # Run bridge (this will download the model on first run)
                # Use longer timeout for initial download
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for download
                )

                if result.returncode == 0:
                    console.print(f"  [green]✓ Model downloaded successfully via bridge[/green]")
                else:
                    console.print(f"  [red]✗ Bridge failed: {result.stderr}[/red]")

            finally:
                # Clean up temp file
                try:
                    Path(temp_path).unlink()
                except:
                    pass

        except subprocess.TimeoutExpired:
            console.print(f"  [red]✗ Bridge download timed out (>10 minutes)[/red]")
        except Exception as e:
            console.print(f"  [red]✗ Failed to trigger bridge download: {e}[/red]")

    async def _trigger_hf_download(self, status: ModelStatus):
        """Trigger HuggingFace model download."""
        if not status.model_info.repo_id:
            return

        try:
            from huggingface_hub import snapshot_download

            console.print(f"  Downloading from {status.model_info.repo_id}...")

            # Download to custom cache if specified (for MLX/FasterWhisper)
            # Otherwise use default cache
            cache_dir = status.model_info.hf_cache_dir if status.model_info.hf_cache_dir else None

            # Download the model
            snapshot_download(
                repo_id=status.model_info.repo_id,
                cache_dir=cache_dir,
            )

            console.print(f"  [green]✓ Downloaded {status.model_info.model_name}[/green]")

        except Exception as e:
            console.print(f"  [red]✗ Failed to download: {e}[/red]")

    async def _trigger_native_download(self, status: ModelStatus):
        """Trigger download using implementation's native load_model() method.

        This is used for implementations like whisper-mps that download from
        non-HuggingFace sources (e.g., openaipublic.azureedge.net).
        """
        try:
            console.print(f"  Downloading via {status.implementation}'s native mechanism...")

            # Find the implementation class by name
            from mac_whisper_speedtest.implementations import get_all_implementations

            impl_class = None
            for impl in get_all_implementations():
                if impl.__name__ == status.implementation:
                    impl_class = impl
                    break

            if not impl_class:
                console.print(f"  [red]✗ Could not find implementation class[/red]")
                return

            # Instantiate and call load_model
            # The implementation's load_model() handles the download automatically
            impl_instance = impl_class()
            impl_instance.load_model(status.model_info.model_name)

            console.print(f"  [green]✓ Downloaded {status.model_info.model_name}.pt via native download[/green]")

        except Exception as e:
            console.print(f"  [red]✗ Failed to trigger native download: {e}[/red]")

    def copy_from_hf_cache(self, statuses: List[ModelStatus]):
        """Copy models from HuggingFace cache to custom cache directories."""
        console.print("\n[bold]Copying models from HuggingFace cache...[/bold]")

        copied_count = 0
        for status in statuses:
            # Only copy if model exists in HF cache but missing from local cache
            if status.hf_cache_status == "complete" and status.local_cache_status == "missing":
                # Only works for implementations using custom HF cache directories
                if status.model_info.hf_cache_dir and status.model_info.repo_id:
                    if self._copy_hf_to_custom_cache(status):
                        copied_count += 1
                else:
                    console.print(f"  {status.implementation}: [yellow]Not applicable (doesn't use custom cache)[/yellow]")

        if copied_count > 0:
            console.print(f"\n[green]✓ Successfully copied {copied_count} model(s)[/green]")
        else:
            console.print("\n[yellow]No models needed to be copied[/yellow]")

    def _copy_hf_to_custom_cache(self, status: ModelStatus) -> bool:
        """Copy a model from default HF cache to custom cache directory.

        Returns:
            True if successful, False otherwise
        """
        import shutil
        from huggingface_hub import scan_cache_dir

        try:
            console.print(f"\n[cyan]Copying {status.implementation} ({status.model_info.repo_id})...[/cyan]")

            # Find the model in default HF cache
            default_cache_info = scan_cache_dir()
            source_path = None

            for repo in default_cache_info.repos:
                if repo.repo_id == status.model_info.repo_id:
                    source_path = repo.repo_path
                    break

            if not source_path:
                console.print(f"  [red]✗ Model not found in default HF cache[/red]")
                return False

            # Construct destination path in custom cache
            # HF cache structure: models--{org}--{model}
            dest_dir = Path(status.model_info.hf_cache_dir)
            repo_folder = f"models--{status.model_info.repo_id.replace('/', '--')}"
            dest_path = dest_dir / repo_folder

            # Check if already exists (shouldn't happen but just in case)
            if dest_path.exists():
                console.print(f"  [yellow]⚠ Already exists at {dest_path}[/yellow]")
                return False

            # Copy the entire repository directory
            console.print(f"  Copying from: {source_path}")
            console.print(f"  To: {dest_path}")

            shutil.copytree(source_path, dest_path, symlinks=True)

            # Verify the copy
            if dest_path.exists():
                size_mb = self._calculate_directory_size(dest_path)
                console.print(f"  [green]✓ Successfully copied {size_mb:.1f} MB[/green]")
                return True
            else:
                console.print(f"  [red]✗ Copy failed - destination not found[/red]")
                return False

        except Exception as e:
            console.print(f"  [red]✗ Copy failed: {e}[/red]")
            import traceback
            console.print(f"  [dim]{traceback.format_exc()}[/dim]")
            return False

    def _calculate_directory_size(self, path: Path) -> float:
        """Calculate total size of directory in MB."""
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024 * 1024)

    async def interactive_menu(self, statuses: List[ModelStatus], model_size: str, implementations: List):
        """Show interactive menu for user actions."""
        while True:
            console.print("\n[bold]What would you like to do?[/bold]")
            console.print("  1. Quit")
            console.print("  2. Copy models from HuggingFace cache")
            console.print("  3. Download all missing models")
            console.print("  4. Select specific implementations to download")

            choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4"], default="1")

            if choice == "1":
                console.print("[green]Exiting...[/green]")
                break
            elif choice == "2":
                self.copy_from_hf_cache(statuses)
            elif choice == "3":
                await self.download_missing_models(statuses, model_size, implementations)
            elif choice == "4":
                await self.interactive_download(statuses, model_size, implementations)

    async def interactive_download(self, statuses: List[ModelStatus], model_size: str, implementations: List):
        """Interactively select implementations to download."""
        # Get implementations with missing models
        missing = [
            s for s in statuses
            if s.local_cache_status in ["missing", "incomplete"]
        ]

        if not missing:
            console.print("[green]All models are already downloaded![/green]")
            return

        console.print("\n[bold]Select implementations to download:[/bold]")
        for i, status in enumerate(missing, 1):
            console.print(f"  {i}. {status.implementation} ({status.model_info.model_name})")

        console.print(f"  {len(missing) + 1}. All of the above")
        console.print(f"  {len(missing) + 2}. Cancel")

        choices = [str(i) for i in range(1, len(missing) + 3)]
        choice = Prompt.ask("Enter your choice", choices=choices, default=str(len(missing) + 2))

        if choice == str(len(missing) + 2):
            console.print("[yellow]Download cancelled[/yellow]")
            return

        if choice == str(len(missing) + 1):
            # Download all
            await self.download_missing_models(statuses, model_size, implementations)
        else:
            # Download selected
            selected_idx = int(choice) - 1
            selected_status = missing[selected_idx]
            implementations_to_download = {selected_status.implementation}
            await self.download_missing_models(statuses, model_size, implementations, implementations_to_download)
