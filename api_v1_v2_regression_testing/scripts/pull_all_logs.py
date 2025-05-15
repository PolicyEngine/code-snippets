import asyncio
import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from google.cloud import logging_v2
from google.cloud.logging_v2 import LogEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = "logs_output"
LOGS_PER_FILE = 50
MAX_CONCURRENT_REQUESTS = 5
RATE_LIMIT_PAUSE_SECONDS = 1
LOG_PREFIX = "CALCULATE_ECONOMY_SIMULATION_JOB"
PROJECT_ID = "your-gcp-project-id"  # Replace with your GCP project ID


class LogDownloader:
    """Handles downloading logs from Google Cloud Logging."""
    
    def __init__(self, project_id: str):
        self.client = logging_v2.Client(project=project_id)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
    async def fetch_logs_page(self, page_token: Optional[str] = None) -> tuple[List[LogEntry], Optional[str]]:
        """Fetch a page of logs matching our filter criteria."""
        async with self.semaphore:
            logger.info(f"Fetching logs page with token: {page_token or 'None'}")
            
            # Define the filter to get logs where jsonPayload.message starts with our prefix
            filter_str = f'resource.type="gae_app" jsonPayload.message="{LOG_PREFIX}*"'
            
            try:
                # Use the logging client to list entries with the filter
                iterator = self.client.list_entries(
                    filter_=filter_str,
                    page_size=100,  # Adjust as needed
                    page_token=page_token
                )
                
                # Get the current page of results
                page = next(iterator.pages)
                entries = list(page)
                next_page_token = iterator.next_page_token
                
                logger.info(f"Fetched {len(entries)} logs. Next page token: {next_page_token or 'None'}")
                
                # Pause to avoid rate limiting
                await asyncio.sleep(RATE_LIMIT_PAUSE_SECONDS)
                
                return entries, next_page_token
                
            except Exception as e:
                logger.error(f"Error fetching logs: {e}")
                raise

    async def download_all_logs(self) -> List[Dict]:
        """Download all logs matching our criteria."""
        all_logs = []
        next_page_token = None
        page_count = 0
        
        logger.info("Starting log download process")
        
        while True:
            page_count += 1
            logger.info(f"Processing page {page_count}")
            
            entries, next_page_token = await self.fetch_logs_page(next_page_token)
            
            # Process each log entry
            for entry in entries:
                # Convert the entry to a dictionary and add to our results
                entry_dict = self._entry_to_dict(entry)
                if entry_dict:
                    all_logs.append(entry_dict)
            
            # If there's no next page token, we've reached the end
            if not next_page_token:
                break
                
        logger.info(f"Download complete. Total logs: {len(all_logs)}")
        return all_logs
        
    def _entry_to_dict(self, entry: LogEntry) -> Optional[Dict]:
        """Convert a log entry to a dictionary, or return None if it doesn't match criteria."""
        try:
            # Ensure the entry has jsonPayload and message
            if not hasattr(entry, 'json_payload') or 'message' not in entry.json_payload:
                return None
                
            # Check if message starts with our prefix
            message = entry.json_payload['message']
            if not isinstance(message, str) or not message.startswith(LOG_PREFIX):
                return None
                
            # Convert to dictionary for JSON serialization
            entry_dict = {
                'timestamp': entry.timestamp.isoformat() if entry.timestamp else None,
                'logName': entry.log_name,
                'severity': entry.severity,
                'jsonPayload': dict(entry.json_payload),
                'resource': {
                    'type': entry.resource.type,
                    'labels': dict(entry.resource.labels)
                },
                'insertId': entry.insert_id,
                'labels': dict(entry.labels) if entry.labels else {}
            }
            
            return entry_dict
            
        except Exception as e:
            logger.error(f"Error processing log entry: {e}")
            return None


class LogDataProcessor:
    """Processes downloaded log data and generates metrics."""
    
    def __init__(self, logs: List[Dict]):
        self.logs = logs
        
    def chunk_logs(self) -> List[List[Dict]]:
        """Split logs into chunks of LOGS_PER_FILE size."""
        return [
            self.logs[i:i + LOGS_PER_FILE] 
            for i in range(0, len(self.logs), LOGS_PER_FILE)
        ]
        
    def calculate_metrics(self) -> Dict:
        """Generate metrics based on the log data."""
        metrics = {
            'total_logs': len(self.logs),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Count occurrences of each message type
        message_counter = Counter()
        severity_counter = Counter()
        logs_per_day = defaultdict(int)
        
        for log in self.logs:
            # Count message types
            message = log.get('jsonPayload', {}).get('message', '')
            if message:
                # Extract the message type (everything after the prefix)
                message_type = message[len(LOG_PREFIX):].strip()
                if message_type:
                    message_counter[message_type] += 1
                else:
                    message_counter['BASE'] += 1
            
            # Count by severity
            severity = log.get('severity')
            if severity:
                severity_counter[severity] += 1
                
            # Count by day
            timestamp = log.get('timestamp')
            if timestamp:
                try:
                    day = timestamp.split('T')[0]
                    logs_per_day[day] += 1
                except Exception:
                    pass
        
        # Add counts to metrics
        metrics['message_type_counts'] = dict(message_counter)
        metrics['severity_counts'] = dict(severity_counter)
        metrics['logs_per_day'] = dict(logs_per_day)
        
        # Add time range metrics
        timestamps = [
            log.get('timestamp') for log in self.logs 
            if log.get('timestamp')
        ]
        
        if timestamps:
            metrics['time_range'] = {
                'oldest_log': min(timestamps),
                'newest_log': max(timestamps)
            }
            
        return metrics


class FileManager:
    """Handles file operations for saving logs and metrics."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._ensure_output_directory()
        
    def _ensure_output_directory(self):
        """Create the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
            
    def save_log_chunks(self, log_chunks: List[List[Dict]]):
        """Save log chunks to numbered JSON files."""
        for i, chunk in enumerate(log_chunks):
            filename = os.path.join(self.output_dir, f"logs_chunk_{i+1}.json")
            self._save_json_file(filename, chunk)
            logger.info(f"Saved log chunk {i+1} with {len(chunk)} logs to {filename}")
            
    def save_metrics(self, metrics: Dict):
        """Save metrics to a JSON file."""
        filename = os.path.join(self.output_dir, "metrics.json")
        self._save_json_file(filename, metrics)
        logger.info(f"Saved metrics to {filename}")
        
    def _save_json_file(self, filename: str, data: Any):
        """Save data to a JSON file with pretty formatting."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


async def main():
    """Main function to orchestrate the log download and processing."""
    start_time = time.time()
    logger.info("Starting log download and processing")
    
    try:
        # Initialize components
        downloader = LogDownloader(project_id=PROJECT_ID)
        
        # Download all logs
        logs = await downloader.download_all_logs()
        logger.info(f"Downloaded {len(logs)} logs matching criteria")
        
        # Process logs
        processor = LogDataProcessor(logs)
        log_chunks = processor.chunk_logs()
        metrics = processor.calculate_metrics()
        
        # Save to files
        file_manager = FileManager(OUTPUT_DIR)
        file_manager.save_log_chunks(log_chunks)
        file_manager.save_metrics(metrics)
        
        # Log summary
        elapsed_time = time.time() - start_time
        logger.info(f"Process completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total logs: {metrics['total_logs']}")
        logger.info(f"Saved in {len(log_chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    asyncio.run(main())