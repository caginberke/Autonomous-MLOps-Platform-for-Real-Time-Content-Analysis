import json
import boto3
import csv
from boto3.dynamodb.conditions import Key
import botocore.exceptions

def get_datas(label):
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table = dynamodb.Table('youtube_comment_analysis')
    items = [] 
    query_params = {
        'IndexName': 'final_label-index',
        'KeyConditionExpression': Key('final_label').eq(label)
    }

    while True:
        response = table.query(**query_params)
        items.extend(response['Items'])
        if 'LastEvaluatedKey' not in response:
            break
        query_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
    
    return items

def lambda_handler(event, context):
    
    labels = [
        'ovgu',
        'elestiri',
        'sikayet',
        'ofke',
        'saka',
        'oneri',
        'soru',
        'eglence',
        'notr'
    ] 
    
    s3 = boto3.resource('s3')
    bucket_name = 'mlops1-train-datas' 
    bucket = s3.Bucket(bucket_name)
    
    key = 'all_labels.csv'
    local_file = '/tmp/datas.csv' 

    existing_rows = []
    
    try:
        bucket.download_file(key, local_file)
        
        with open(local_file, 'r', encoding='utf-8') as infile:  
            reader = csv.reader(infile)
            existing_rows = list(reader)
            
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['404', '403']:
            if error_code == '403':
                print(f"Warning: 403 Forbidden on S3 download for {key}. Treating as non-existent.")
            pass
        else:
            raise
    
    if len(existing_rows) == 0:
        existing_rows.append(['final_label', 'message'])
    
    all_new_items = []
    
    for label in labels:
        try:
            new_items = get_datas(label)
            print(f"{label}: {len(new_items)} kayıt bulundu")
            all_new_items.extend(new_items)
        except Exception as e:
            print(f"{label} için hata: {str(e)}")
            continue
    
    for item in all_new_items:
        row = [
            item.get('final_label', ''),
            item.get('message', '')
        ]
        existing_rows.append(row)
    
    with open(local_file, 'w', newline='', encoding='utf-8') as outfile: 
        writer = csv.writer(outfile)
        writer.writerows(existing_rows)
    
    bucket.upload_file(local_file, key)
    
    s3_path = f's3://{bucket_name}/{key}'

    return {
        'statusCode': 200,
        'body': json.dumps({
            's3_path': s3_path,
            'total_rows': len(existing_rows) - 1,
            'new_items': len(all_new_items)
        })
    }