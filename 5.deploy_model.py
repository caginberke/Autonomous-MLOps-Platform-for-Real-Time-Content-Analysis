import json
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

def lambda_handler(event, context):

    try:
        
        if 'detail' in event:
            training_job_name = event['detail']['TrainingJobName']
        elif 'TrainingJobName' in event:
            training_job_name = event['TrainingJobName']
        else:
            raise ValueError("Training job name bulunamadı")
        
        sagemaker = boto3.client('sagemaker', region_name='eu-north-1')
        
        training_job = sagemaker.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        model_artifact = training_job['ModelArtifacts']['S3ModelArtifacts']
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f'mlops1-blazingtext-model-{timestamp}'
        endpoint_config_name = f'mlops1-blazingtext-config-{timestamp}'
        endpoint_name = 'mlops1-blazingtext-endpoint'
        
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '669576153137.dkr.ecr.eu-north-1.amazonaws.com/blazingtext:latest',
                'ModelDataUrl': model_artifact,
                'Mode': 'SingleModel'
            },
            ExecutionRoleArn=''
        )
        
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1
                }
            ]
        )
        
        try:
            sagemaker.describe_endpoint(EndpointName=endpoint_name)
            
            sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            action = 'updated'
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                sagemaker.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
                action = 'created'
            else:
                raise
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Model successfully deployed and endpoint {action}',
                'training_job_name': training_job_name,
                'model_name': model_name,
                'endpoint_config_name': endpoint_config_name,
                'endpoint_name': endpoint_name,
                'model_artifact': model_artifact,
                'action': action
            })
        }
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to deploy model'
            })
        }