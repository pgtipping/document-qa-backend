from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime
import boto3
import json
from botocore.exceptions import ClientError
from app.core.config import settings
from app.core.security import get_api_key


router = APIRouter()


@router.get("")
async def get_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: str = Depends(get_api_key)
) -> List[dict]:
    """
    Fetch performance metrics from S3.
    Optionally filter by date range and model.
    """
    try:
        if not settings.S3_BUCKET:
            raise HTTPException(
                status_code=500,
                detail="S3 storage not configured"
            )

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )

        # List objects in the metrics prefix
        prefix = settings.S3_PERFORMANCE_LOGS_PREFIX
        response = s3_client.list_objects_v2(
            Bucket=settings.S3_BUCKET,
            Prefix=prefix
        )

        metrics = []
        for obj in response.get('Contents', []):
            # Parse timestamp from key
            key = obj['Key']
            try:
                # Extract timestamp from key
                # Format: prefix/YYYY-MM-DD_HH-MM-SS_docid.json
                timestamp_str = key.split('/')[-1].split('_')[0:2]
                timestamp = datetime.strptime(
                    '_'.join(timestamp_str),
                    '%Y-%m-%d_%H-%M-%S'
                )
                
                # Apply date filters if provided
                if start_date:
                    start = datetime.fromisoformat(start_date)
                    if timestamp < start:
                        continue
                if end_date:
                    end = datetime.fromisoformat(end_date)
                    if timestamp > end:
                        continue

                # Get the metric data
                obj_response = s3_client.get_object(
                    Bucket=settings.S3_BUCKET,
                    Key=key
                )
                metric_data = json.loads(obj_response['Body'].read())

                # Apply model filter if provided
                if model and metric_data.get('model') != model:
                    continue

                metrics.append(metric_data)

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"Error processing metric {key}: {str(e)}")
                continue

        # Sort by timestamp
        metrics.sort(key=lambda x: x['timestamp'])
        return metrics

    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch metrics from S3: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        ) 