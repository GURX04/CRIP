"""
remediation/deploy_lambda.py
============================
Automates building and deploying the remediation Lambda to LocalStack.

Uses boto3 to:
1. Package remediation_handler.py into function.zip.
2. Ensure IAM role exists in LocalStack.
3. Create / update Lambda function `crip-cost-remediator`.
4. Create Lambda Function URL (auth-type NONE).
5. Grant public invoke permissions.
6. Print the Function URL ready for anomaly_detection.py webhook integration.
"""

import io
import json
import os
import sys
import zipfile
from pathlib import Path
import boto3
from botocore.config import Config

LOCALSTACK_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
FUNCTION_NAME = "crip-cost-remediator"
ROLE_NAME = "crip-lambda-execution-role"

boto_config = Config(
    region_name=REGION,
    retries={"max_attempts": 3, "mode": "standard"}
)

session = boto3.Session(
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name=REGION
)

iam = session.client("iam", endpoint_url=LOCALSTACK_ENDPOINT, config=boto_config)
lambda_client = session.client("lambda", endpoint_url=LOCALSTACK_ENDPOINT, config=boto_config)


def create_zip_package() -> bytes:
    """Create in-memory zip file containing remediation_handler.py"""
    handler_path = Path(__file__).parent / "remediation_handler.py"
    if not handler_path.exists():
        raise FileNotFoundError(f"Missing {handler_path}")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_path, arcname="remediation_handler.py")

    # Also save to disk for aws-cli usage
    disk_zip = Path(__file__).parent / "function.zip"
    with open(disk_zip, "wb") as f:
        f.write(buffer.getvalue())

    print(f"[*] Packaged {handler_path.name} -> {disk_zip.resolve()} ({disk_zip.stat().st_size} bytes)")
    return buffer.getvalue()


def get_or_create_role() -> str:
    """Create or retrieve a dummy IAM role in LocalStack."""
    assume_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        resp = iam.get_role(RoleName=ROLE_NAME)
        role_arn = resp["Role"]["Arn"]
        print(f"[*] Found existing IAM Role: {role_arn}")
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        pass

    resp = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(assume_policy),
        Description="IAM role for CRIP cost anomaly remediation lambda in LocalStack",
    )
    role_arn = resp["Role"]["Arn"]
    print(f"[+] Created IAM Role: {role_arn}")
    return role_arn


def deploy_function(zip_bytes: bytes, role_arn: str) -> str:
    """Create or update the Lambda function and return its ARN."""
    try:
        lambda_client.get_function(FunctionName=FUNCTION_NAME)
        print(f"[*] Function '{FUNCTION_NAME}' exists. Updating code...")
        resp = lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_bytes,
        )
        return resp["FunctionArn"]
    except lambda_client.exceptions.ResourceNotFoundException:
        pass

    print(f"[+] Creating Lambda function '{FUNCTION_NAME}'...")
    resp = lambda_client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime="python3.11",
        Role=role_arn,
        Handler="remediation_handler.lambda_handler",
        Code={"ZipFile": zip_bytes},
        Timeout=15,
        MemorySize=128,
        Description="Automated dry-run cost remediation handler for CRIP",
    )
    return resp["FunctionArn"]


def get_or_create_function_url() -> str:
    """Create Lambda Function URL and return the HTTP URL."""
    try:
        url_resp = lambda_client.get_function_url_config(FunctionName=FUNCTION_NAME)
        func_url = url_resp["FunctionUrl"]
        print(f"[*] Found existing Function URL: {func_url}")
        return func_url
    except lambda_client.exceptions.ResourceNotFoundException:
        pass

    print(f"[+] Creating Function URL for '{FUNCTION_NAME}'...")
    url_resp = lambda_client.create_function_url_config(
        FunctionName=FUNCTION_NAME,
        AuthType="NONE",
    )
    func_url = url_resp["FunctionUrl"]

    # Grant public invoke permission
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="FunctionURLAllowPublicAccess",
            Action="lambda:InvokeFunctionUrl",
            Principal="*",
            FunctionUrlAuthType="NONE",
        )
        print("[+] Added public invoke permission for Function URL.")
    except Exception as e:
        print(f"[!] Permission grant note: {e}")

    return func_url


def main():
    print("=" * 65)
    print("  CRIP Remediation Lambda Deployment to LocalStack")
    print(f"  Endpoint : {LOCALSTACK_ENDPOINT}")
    print("=" * 65)

    try:
        zip_bytes = create_zip_package()
        role_arn = get_or_create_role()
        fn_arn = deploy_function(zip_bytes, role_arn)
        func_url = get_or_create_function_url()

        print("-" * 65)
        print(f"  [SUCCESS] Lambda Deployed: {FUNCTION_NAME}")
        print(f"  [ARN]     {fn_arn}")
        print(f"  [WEBHOOK] {func_url}")
        print("=" * 65)
        print("\nNext step: Run anomaly detection with webhook integration:")
        print(f"  python anomaly_detection.py --from-csv reports/synthetic_cur.csv --webhook-url \"{func_url}\"")
        print()

    except Exception as err:
        print(f"\n[ERROR] Deployment failed: {err}")
        print("Ensure LocalStack container is running: docker compose up -d localstack")
        sys.exit(1)


if __name__ == "__main__":
    main()
