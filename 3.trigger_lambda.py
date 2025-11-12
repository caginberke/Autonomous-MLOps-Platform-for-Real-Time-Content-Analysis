import json
import boto3
import datetime
import pandas as pd
import io
import traceback

def lambda_handler(event, context):
    try:
        if "body" in event:
            body = json.loads(event.get("body", "{}"))
        else:
            body = event  
        
        s3_path = body.get("s3_path", "")
        
        if not s3_path:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 's3_path is required'})
            }
        
        if not s3_path.startswith('s3://'):
            raise ValueError("s3_path must start with 's3://'")
        parts = s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        if not key:
            raise ValueError("Invalid s3_path: missing key")
        
        s3_client = boto3.client('s3')
        
        csv_obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(csv_obj['Body'].read()), encoding='utf-8')
        
        required_cols = ['final_label', 'message']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns: {required_cols}")
        
        blazingtext_content = ""
        for _, row in df.iterrows():
            blazingtext_content += f"__label__{row['final_label']} {row['message']}\n"
        
        formatted_key = "formatted/all_blazingtext.txt"
        s3_client.put_object(
            Bucket=bucket,
            Key=formatted_key,
            Body=blazingtext_content.encode('utf-8')
        )
        
        formatted_s3_path = f"s3://{bucket}/{formatted_key}"
        output_s3_path = f"s3://{bucket}/models/"
        
        sagemaker_client = boto3.client('sagemaker', region_name='eu-north-1')
        
        algorithm_uri = '669576153137.dkr.ecr.eu-north-1.amazonaws.com/blazingtext:latest'
        
        training_job_name = f'Training-all-labels-{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}'
        
        sagemaker_client.create_training_job(
            TrainingJobName=training_job_name,
            AlgorithmSpecification={
                'TrainingImage': algorithm_uri,
                'TrainingInputMode': 'File'
            },
            RoleArn='',
            HyperParameters={
                'mode': 'supervised',
                'epochs': '10',
                'learning_rate': '0.05',
                'word_ngrams': '2',
                'vector_dim': '10',
                'min_count': '1',
                'early_stopping': 'False'
            },
            InputDataConfig=[
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': formatted_s3_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/plain',
                    'CompressionType': 'None'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': output_s3_path
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 5
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 600
            }
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Training job started successfully',
                'training_job_name': training_job_name,
                'original_file': s3_path,
                'formatted_file': formatted_s3_path,
                's3_output': output_s3_path
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Failed to start training job'
            })
        }