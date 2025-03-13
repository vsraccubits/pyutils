import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class HuggingFaceDownloader:
    def __init__(self, model_name: str, token: Optional[str] = None):
        """Initialize the HuggingFaceDownloader."""
        self.model_name = model_name
        self.token = token
        self.api = HfApi(token=token)

    def get_file_size(self, filename: str) -> int:
        """Get file size in bytes from Hugging Face."""
        try:
            url = hf_hub_url(repo_id=self.model_name, filename=filename)
            hf_file_metadata = get_hf_file_metadata(url=url, token=self.token)
            return hf_file_metadata.size
        except Exception as e:
            logger.error("Error getting file size for %s: %s", filename, e)
            return 0

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Convert size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def list_repository_files(self, debug: bool = False) -> List[Dict]:
        """List all files in a Hugging Face repository with their sizes.

        Args:
            debug (bool): Whether to display the formatted output

        Returns:
            List[Dict]: List of dictionaries containing file information
        """
        try:
            files = self.api.list_repo_files(self.model_name)

            # Get file information
            file_info = []
            total_size = 0

            # Process each file
            for index, file in enumerate(files, 1):
                size = self.get_file_size(file)
                path = Path(file)

                file_info.append({
                    'index': index,
                    'filename': path.name,
                    'path': str(path.parent),
                    'size_bytes': size,
                    'size_formatted': self.format_size(size)
                })

                total_size += size

            if debug:
                print(f"Repository: {self.model_name}")
                print(f"Total Files: {len(files)}")
                print(f"Total Size: {self.format_size(total_size)}")
                print("\nFile Listing:")
                print("-" * 80)
                print(f"{'#':4} {'File Name':30} {'Size':10} {'Path':30}")
                print("-" * 80)

                # Print file information
                for file in file_info:
                    print(f"{file['index']:<4} {file['filename'][:30]:30} "
                        f"{file['size_formatted']:10} "
                        f"{file['path'] if file['path'] != '.' else '/':<30}")
                print("-" * 80)

            return file_info

        except Exception as e:
            logger.error("Error listing repository files: %s", e)
            raise

    def download_file(self, filename: str, output_dir: str) -> str:
        """Download individual file from Hugging Face repository.

        Args:
            filename (str): Name of the file to download
            output_dir (str): Directory where file should be downloaded
            token (str, optional): Hugging Face API token for private repos

        Returns:
            str: Path to the downloaded file
        """
        try:
            logger.debug("Downloading file: %s", filename)

            # Download the file
            local_path = hf_hub_download(
                repo_id=self.model_name,
                filename=filename,
                local_dir=output_dir,
                token=self.token,
                local_dir_use_symlinks=False
            )

            logger.debug("File downloaded successfully to: %s", local_path)
            return local_path

        except Exception as e:
            logger.error("Error downloading file %s: %s", filename, e)
            raise

    def download_repository_files(self, output_dir: str) -> None:
        """List and download files individually from a Hugging Face repository.

        Args:
            output_dir (str): Directory where files should be downloaded
            token (str, optional): Hugging Face API token for private repos
        """
        try:
            # First, list all files and show information
            files = self.list_repository_files(debug=True)

            # Ask for confirmation before downloading
            total_size = sum(file['size_bytes'] for file in files)
            logger.debug("Total download size will be: %s", self.format_size(total_size))

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Download each file individually
            # for file_info in tqdm(files, desc="Downloading files"):
            for file_info in files:
                network_speed_bps = self.calculate_download_speed()
                network_speed_mbps = network_speed_bps / (1024 * 1024)
                logger.info("Network speed: %s MB/s", network_speed_mbps)
                eta_seconds = self.calculate_eta(file_info['size_bytes'], network_speed_bps)
                formatted_eta = self.format_eta(eta_seconds)
                logger.info("ETA for %s: %s, %s", file_info['filename'], formatted_eta, file_info['size_formatted'])
                filename = file_info['path'] + '/' + file_info['filename'] if file_info['path'] != '.' else file_info['filename']
                file_size = file_info['size_formatted']

                logger.debug("Starting download of %s (%s)", filename, file_size)
                try:
                    self.download_file(
                        filename=filename,
                        output_dir=output_dir
                    )
                    logger.debug("Successfully downloaded: %s", filename)
                except Exception as e:
                    logger.error("Failed to download %s: %s", filename, e)

            logger.debug("Download completed. Files saved in: %s", output_dir)

        except Exception as e:
            logger.error("Error in download operation: %s", e)

    @staticmethod
    def calculate_download_speed(chunk_size=1024, test_duration=2):
        """Calculate download speed in bytes per second."""
        URL = "https://huggingface.co"
        try:
            response = requests.get(URL, stream=True)
            response.raise_for_status()

            start_time = time.time()
            total_bytes = 0

            # Measure download speed for a few seconds
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_bytes += len(chunk)
                elapsed_time = time.time() - start_time
                if elapsed_time > test_duration:
                    break

            # Calculate speed
            speed_bps = total_bytes / elapsed_time

            return speed_bps

        except requests.exceptions.RequestException as e:
            logger.error("Error: %s", e)
            return 0

    @staticmethod
    def calculate_eta(file_size_bytes: int, speed_bps: float) -> float:
        """Calculate estimated download time in seconds.

        Args:
            file_size_bytes (int): Size of file in bytes
            speed_bps (float): Download speed in bytes per second

        Returns:
            float: Estimated time in seconds, or inf if speed is 0
        """
        if speed_bps <= 0:
            return float('inf')

        return file_size_bytes / speed_bps

    @staticmethod
    def format_eta(eta_seconds: float) -> str:
        """Format ETA into human readable string with completion time.

        Args:
            eta_seconds (float): Estimated time in seconds

        Returns:
            str: Formatted string with ETA and completion time
        """
        if eta_seconds == float('inf'):
            return "Unknown"

        # Format time duration
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.1f} seconds"
        elif eta_seconds < 3600:
            minutes = eta_seconds / 60
            eta_str = f"{minutes:.1f} minutes"
        else:
            hours = eta_seconds / 3600
            eta_str = f"{hours:.1f} hours"

        return eta_str


if __name__ == "__main__":
    downloader = HuggingFaceDownloader(model_name="HuggingFaceH4/zephyr-7b-beta", token="")
    downloader.download_repository_files(output_dir="./downloaded_files")
