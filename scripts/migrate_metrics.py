import json
import boto3  # type: ignore
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Match
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


def parse_markdown_metrics(content: str) -> List[Dict[str, Any]]:
    """Parse metrics from markdown format."""
    metrics = []
    entries = content.split('---')
    
    for entry in entries:
        if not entry.strip():
            continue
            
        try:
            # Extract timestamp
            timestamp_match: Optional[Match[str]] = re.search(
                r'\*\*Timestamp:\*\* (.*)', 
                entry
            )
            timestamp = timestamp_match.group(1) if timestamp_match else None
            
            # Extract model and question
            model_match: Optional[Match[str]] = re.search(
                r'\*\*Model:\*\* (.*)',
                entry
            )
            model = model_match.group(1) if model_match else None
            
            question_match: Optional[Match[str]] = re.search(
                r'\*\*Question:\*\* (.*)',
                entry
            )
            question = question_match.group(1) if question_match else None
            
            # Extract document metrics
            size_match = re.search(
                r'\*\*Size:\*\* ([\d.]+) KB',
                entry
            )
            if not size_match:
                continue
                
            chunks_match = re.search(
                r'\*\*Total Chunks:\*\* (\d+)',
                entry
            )
            if not chunks_match:
                continue
                
            selected_match = re.search(
                r'\*\*Selected Chunks:\*\* (\d+)',
                entry
            )
            if not selected_match:
                continue
                
            chunk_size_match = re.search(
                r'\*\*Chunk Size:\*\* (\d+)',
                entry
            )
            if not chunk_size_match:
                continue
                
            context_match = re.search(
                r'\*\*Context Length:\*\* (\d+)',
                entry
            )
            if not context_match:
                continue
                
            doc_metrics = {
                'size_kb': float(size_match.group(1)),
                'total_chunks': int(chunks_match.group(1)),
                'selected_chunks': int(selected_match.group(1)),
                'chunk_size': int(chunk_size_match.group(1)),
                'context_length': int(context_match.group(1))
            }
            
            # Extract LLM timings
            llm_section = re.search(
                r'### LLM Processing Time Breakdown.*?### ',
                entry,
                re.DOTALL
            )
            if not llm_section:
                continue
                
            llm_text = llm_section.group(0)
            time_pattern = r'\*\*Total Time:\*\* ([\d.]+)s'
            llm_time_match = re.search(time_pattern, llm_text)
            if not llm_time_match:
                continue
                
            llm_total = float(llm_time_match.group(1))
            llm_timings = []
            timing_pattern = (
                r'- ([^:]+): ([\d.]+)s \(([\d.]+)%\)'
            )
            for timing in re.finditer(timing_pattern, llm_text):
                llm_timings.append({
                    'name': timing.group(1),
                    'value': float(timing.group(2)),
                    'percentage': float(timing.group(3))
                })
            
            # Extract document timings
            doc_pattern = (
                r'### Document Processing Time Breakdown.*?'
                r'(?=---|$)'
            )
            doc_section = re.search(
                doc_pattern,
                entry,
                re.DOTALL
            )
            if not doc_section:
                continue
                
            doc_text = doc_section.group(0)
            doc_time_match = re.search(time_pattern, doc_text)
            if not doc_time_match:
                continue
                
            doc_total = float(doc_time_match.group(1))
            doc_timings = []
            for timing in re.finditer(timing_pattern, doc_text):
                doc_timings.append({
                    'name': timing.group(1),
                    'value': float(timing.group(2)),
                    'percentage': float(timing.group(3))
                })
            
            metrics.append({
                'timestamp': timestamp,
                'model': model,
                'provider': model.split(' ')[-1].strip('()') if model else None,
                'question': question,
                'document_metrics': doc_metrics,
                'llm_timing': llm_timings,
                'doc_timing': doc_timings,
                'total_llm_time': llm_total,
                'total_doc_time': doc_total
            })
            
        except (AttributeError, ValueError) as e:
            print(f"Error parsing entry: {str(e)}")
            continue
            
    return metrics


def read_local_metrics(file_path: str) -> List[Dict[str, Any]]:
    """Read metrics from local file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        if file_path.endswith('.json'):
            return json.loads(content)
        elif file_path.endswith('.md'):
            return parse_markdown_metrics(content)
        else:
            print(f"Unsupported file format: {file_path}")
            return []
            
    except FileNotFoundError:
        print(f"No metrics file found at {file_path}")
        return []
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing metrics file {file_path}: {str(e)}")
        return []


def upload_to_s3(
    metrics: List[Dict[str, Any]],
    bucket: str,
    prefix: str
) -> None:
    """Upload metrics to S3."""
    if not metrics:
        print("No metrics to upload")
        return

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

    # Upload each metric entry as a separate file
    for metric in metrics:
        timestamp = metric.get(
            'timestamp',
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )
        doc_id = metric.get('document_id', 'unknown')
        key = f"{prefix}{timestamp}_{doc_id}.json"
        
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(metric, indent=2),
                ContentType='application/json'
            )
            print(f"Uploaded metric to {key}")
        except Exception as e:
            print(f"Failed to upload metric {key}: {str(e)}")


def main():
    # Load environment variables
    bucket = os.getenv('S3_BUCKET')
    prefix = os.getenv('S3_PERFORMANCE_LOGS_PREFIX', 'metrics/')
    
    if not bucket:
        print("S3_BUCKET environment variable not set")
        return

    # Paths to check for metrics files
    paths = [
        'performance_logs/performance_metrics.md',
        '../document-qa-frontend/performance_logs/performance_metrics.json',
        'performance_logs/performance_metrics.json',
    ]

    for path in paths:
        print(f"Checking {path}...")
        metrics = read_local_metrics(path)
        if metrics:
            print(f"Found {len(metrics)} metrics in {path}")
            upload_to_s3(metrics, bucket, prefix)
            print("Migration complete!")
            return

    print("No metrics files found in any of the expected locations")


if __name__ == '__main__':
    main() 