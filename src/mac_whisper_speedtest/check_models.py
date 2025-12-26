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

    def __init__(self):
        self.hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

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
                    hf_status, hf_size = self._check_hf_cache(model_info)

                    # Check local cache
                    local_status, local_size = self._check_local_cache(model_info)

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

    def _check_hf_cache(self, model_info: ModelInfo) -> tuple[str, Optional[float]]:
        """Check if model exists in HuggingFace cache.

        Returns:
            Tuple of (status, size_in_mb)
            Status: "complete", "missing", "incomplete", or "n/a"
        """
        if not model_info.repo_id:
            return "n/a", None

        if model_info.verification_method == "huggingface":
            return self._verify_hf_model(model_info.repo_id)

        return "n/a", None

    def _verify_hf_model(self, repo_id: str) -> tuple[str, Optional[float]]:
        """Verify HuggingFace model using huggingface_hub tools.

        Returns:
            Tuple of (status, size_in_mb)
        """
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()

            # Find the repo in cache
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    # Calculate total size
                    total_size = repo.size_on_disk
                    size_mb = total_size / (1024 * 1024)

                    # For now, assume complete if it exists
                    # TODO: Could verify against remote metadata
                    return "complete", size_mb

            return "missing", None

        except ImportError:
            log.warning("huggingface_hub not available for cache verification")
            return "n/a", None
        except Exception as e:
            log.error(f"Error verifying HF model {repo_id}: {e}")
            return "incomplete", None

    def _check_local_cache(self, model_info: ModelInfo) -> tuple[str, Optional[float]]:
        """Check if model exists in local cache paths.

        Returns:
            Tuple of (status, size_in_mb)
        """
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

        # Verify size if expected size is provided
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
        table.add_column("Size (MB)", justify="right")

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
        ready = sum(1 for s in statuses if s.local_cache_status == "complete")
        missing = sum(1 for s in statuses if s.local_cache_status == "missing")
        incomplete = sum(1 for s in statuses if s.local_cache_status == "incomplete")

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

            # Download the model
            snapshot_download(
                repo_id=status.model_info.repo_id,
                local_dir=None,  # Uses default cache
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
        """Copy models from HuggingFace cache to local caches."""
        # This is mainly for FluidAudio which needs models copied to Application Support
        console.print("\n[bold]Copying models from HuggingFace cache...[/bold]")

        for status in statuses:
            if status.hf_cache_status == "complete" and status.local_cache_status in ["missing", "incomplete"]:
                console.print(f"  {status.implementation}: [yellow]Copy not yet implemented[/yellow]")

        console.print("[green]Copy operation complete[/green]")

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
