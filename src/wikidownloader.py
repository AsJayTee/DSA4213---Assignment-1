"""
Wikidump Downloader
Downloads wikidumps from Archive.org
Retrieves links from config/wikidownloader.json
"""

import time
import json
import logging
import requests
from pathlib import Path
from typing import Generator
from urllib.parse import urlparse
from datetime import datetime, timedelta


class WikiConfig:
    """Retrieves download links from config"""
    
    def __init__(self, config_path: str, logger: logging.Logger):
        self.logger = logger
        self.links: dict[str, str] = {}
        
        path = self.__access_config(config_path)
        self.__retrieve_config(path)

    def get_links(self) -> Generator[tuple[str, str], None, None]:
        """Generator that yields (filename, url) pairs"""
        for filename, url in self.links.items():
            yield filename, url

    def __access_config(self, config_path: str) -> Path:
        """Creates path and checks file exists"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return path
    
    def __retrieve_config(self, path: Path) -> None:
        """Reads file with exception handling"""
        try:
            with path.open("r", encoding="utf-8") as file:
                self.links = json.load(file)  
            self.logger.info(f"Loaded {len(self.links)} links")
        except Exception as e:
            self.logger.error(f"An Unexpected Error has Occurred: {e}")
            raise


class ProgressTracker:
    """Track and report progress with ETA calculation"""
    
    def __init__(self, total_size: int, logger: logging.Logger, unit: str = "bytes"):
        self.total_size = total_size
        self.processed = 0
        self.unit = unit
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.report_interval = 5
        self.logger = logger
    
    def update(self, amount: int) -> None:
        """Update progress and report if needed"""
        self.processed += amount
        current_time = time.time()
        
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
    
    def report(self) -> None:
        """Report current progress with percentage and ETA"""
        if self.total_size > 0:
            percentage = (self.processed / self.total_size) * 100
            elapsed = time.time() - self.start_time
            
            if self.processed > 0:
                rate = self.processed / elapsed
                remaining = (self.total_size - self.processed) / rate if rate > 0 else 0
                eta = datetime.now() + timedelta(seconds=remaining)
                
                self.logger.info(
                    f"Progress: {percentage:.1f}% "
                    f"({self.__format_size(self.processed)}/{self.__format_size(self.total_size)}) "
                    f"Speed: {self.__format_size(rate)}/s "
                    f"ETA: {eta.strftime('%H:%M:%S')}"
                )
            else:
                self.logger.info(f"Progress: {percentage:.1f}%")
    
    def __format_size(self, size: float) -> str:
        """Format size with appropriate unit"""
        if self.unit == "bytes":
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f}{unit}"
                size /= 1024.0
            return f"{size:.1f}TB"
        return f"{size:.1f} {self.unit}"


class DownloadManager:
    """Download manager for Wikimedia Dumps"""
    
    def __init__(self, base_dir: str, logger: logging.Logger):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.download_chunk_size = 8192 * 16
        self.logger = logger
    
    def download_file(self, url: str, base_filename: str) -> Path:
        """
        Download a file from the given URL with progress tracking.
        Supports resuming partially downloaded files.
        """
        extension = Path(urlparse(url).path).suffix
        filename = f"{base_filename}{extension}"
        file_path = self.base_dir / filename
        temp_path = file_path.with_suffix(".part")

        # Check if file already fully downloaded
        if file_path.exists():
            self.logger.info(f"File already exists. Skipping download: {file_path}")
            return file_path

        # Determine starting point for resuming
        resume_byte_pos = temp_path.stat().st_size if temp_path.exists() else 0

        headers = {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos > 0 else {}
        self.logger.info(f"{'Resuming' if resume_byte_pos > 0 else 'Starting'} download: {url}")

        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos
            
            # Pass the logger to the progress tracker
            tracker = ProgressTracker(total_size, self.logger)
            tracker.processed = resume_byte_pos  # Start from existing bytes

            with open(temp_path, 'ab') as f:
                for chunk in response.iter_content(chunk_size=self.download_chunk_size):
                    if chunk:
                        f.write(chunk)
                        tracker.update(len(chunk))

        # Rename temp file to final file
        temp_path.rename(file_path)
        self.logger.info(f"Download completed: {file_path}")
        return file_path


class WikiDownloader:
    """Main downloader class that orchestrates the download process"""
    
    def __init__(self, 
                 config_path: str,
                 base_dir: str,
                 logger: logging.Logger):
        self.logger = logger
        self.wiki_config = WikiConfig(config_path, logger=self.logger)
        self.download_manager = DownloadManager(base_dir, logger=self.logger)

    def download_all(self) -> Generator[Path, None, None]:
        """Downloads files one by one and yields the path after each download."""
        for filename, url in self.wiki_config.get_links():
            file_path = self.download_manager.download_file(url, filename)
            yield file_path


if __name__ == "__main__":
    # Configure default logger for standalone execution
    logger = logging.getLogger("wikidump_downloader")
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler('wikidownloader.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent logging from propagating to the root logger
    logger.propagate = False
    
    # Run downloader with default configuration
    downloader = WikiDownloader(
        config_path="config/wikidownloader.json",
        base_dir="data", 
        logger=logger
    )
    
    for file_path in downloader.download_all():
        logger.info(f"Downloaded: {file_path}")