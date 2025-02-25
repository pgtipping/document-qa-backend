import os
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
from app.core.config import settings
from app.core.logging import logger

class DocumentStorage:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME
        # Use a prefix to organize files within the bucket
        self.prefix = "Doc-Chat"

    async def save_document(self, file: UploadFile, document_id: str) -> str:
        """
        Save a document to S3.
        Returns the S3 URL of the saved document.
        """
        try:
            # Create a unique S3 key with prefix
            s3_key = f"{self.prefix}/documents/{document_id}/{file.filename}"
            
            # Upload the file to S3
            await self._upload_fileobj(file, s3_key)
            
            # Generate the S3 URL
            url = (
                f"https://{self.bucket_name}.s3.{settings.AWS_REGION}"
                f".amazonaws.com/{s3_key}"
            )
            
            logger.info(f"Document saved to S3: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Error saving document to S3: {str(e)}")
            raise

    async def get_document(self, document_id: str, filename: str) -> bytes:
        """
        Retrieve a document from S3.
        """
        try:
            s3_key = f"{self.prefix}/documents/{document_id}/{filename}"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response['Body'].read()
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"Document not found in S3: {s3_key}")
                raise FileNotFoundError(f"Document not found: {filename}")
            raise

    async def delete_document(self, document_id: str, filename: str) -> None:
        """
        Delete a document from S3.
        """
        try:
            s3_key = f"{self.prefix}/documents/{document_id}/{filename}"
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Document deleted from S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error deleting document from S3: {str(e)}")
            raise

    async def _upload_fileobj(self, file: UploadFile, s3_key: str) -> None:
        """
        Helper method to upload a file object to S3.
        """
        try:
            # Read the file content
            content = await file.read()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=file.content_type
            )
            
            # Reset the file pointer for potential future reads
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise 