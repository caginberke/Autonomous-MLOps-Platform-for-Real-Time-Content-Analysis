import json
import boto3

def lambda_handler(event, context):
    dynamodb = boto3.client('dynamodb', region_name='eu-north-1')
    stepfunctions = boto3.client('stepfunctions', region_name='eu-north-1')
    
    new_comments = sum(1 for record in event['Records'] if record['eventName'] == 'INSERT')
    
    if new_comments == 0:
        return {'statusCode': 200, 'body': json.dumps('No new comments to process.')}
    
    try:
        response = dynamodb.update_item(
            TableName='mlops1_counters',  
            Key={'counter_name': {'S': 'comment_count'}},
            UpdateExpression='ADD current_value :inc',
            ExpressionAttributeValues={':inc': {'N': str(new_comments)}},
            ReturnValues='UPDATED_NEW'
        )
        
        updated_count = int(response['Attributes']['current_value']['N'])
        
        if updated_count >= 1000: 
            stepfunctions.start_execution(
                stateMachineArn='',
                input='{}'  
            )
            
            dynamodb.update_item(
                TableName='mlops1_counters',
                Key={'counter_name': {'S': 'comment_count'}},
                UpdateExpression='SET current_value = :zero',
                ExpressionAttributeValues={':zero': {'N': '0'}}
            )
        
        return {'statusCode': 200, 'body': json.dumps(f'Updated count: {updated_count}')}
    
    except Exception as e:
        print(f"Hata: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f'Hata: {str(e)}')}