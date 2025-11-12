import json
from google import genai
import boto3
from botocore.exceptions import ClientError
import requests
import uuid
import time
from decimal import Decimal

CONFIDENCE_THRESHOLD = 0.3
SAGEMAKER_CONFIDENCE_THRESHOLD = 0.9
LABELS = ['övgü', 'eleştiri', 'şikayet', 'öfke', 'şaka', 'öneri', 'soru', 'eğlence', 'nötr']

def transliterate(text):
    mapping = {
        'ö': 'o', 'ü': 'u', 'ğ': 'g', 'ş': 's', 'ı': 'i', 'ç': 'c',
        'Ö': 'O', 'Ü': 'U', 'Ğ': 'G', 'Ş': 'S', 'I': 'I', 'Ç': 'C'
    }
    return ''.join(mapping.get(c, c) for c in text)

def get_geminiapi():
    secret_name = "mlops1/geminiapikey"
    region_name = "eu-north-1"
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    secret = get_secret_value_response["SecretString"]
    return json.loads(secret)["GEMINI_API_KEY"] if "GEMINI_API_KEY" in json.loads(secret) else secret

def get_colab_kumru():
    secret_name = "mlops1/colab_kumru"
    region_name = "eu-north-1"
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    secret = get_secret_value_response['SecretString']
    return json.loads(secret)["COLAB_KUMRU"] if "COLAB_KUMRU" in json.loads(secret) else secret

def get_gemini_label(user_message):
    api_key = get_geminiapi()
    labels_str = ', '.join(LABELS)
    
    prompt = f"""Yorum: "{user_message}"
        Kategoriler: {labels_str}
        SADECE kategori ismini yaz."""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.0-flash-exp", contents=prompt)
    
    gemini_raw = response.text.strip().lower()
    
    for label in LABELS:
        if transliterate(label).lower() in gemini_raw:
            return transliterate(label)
    
    return 'notr'

def get_kumru_message(user_message):
    url = get_colab_kumru()
    myobj = {'text': user_message}
    response = requests.post(url, json=myobj)
    return response.text

def get_blazingtext_data(user_message):
    try:
        payload = {"instances": [user_message]}
        sagemaker_runtime = boto3.client('runtime.sagemaker', region_name='eu-north-1')
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='mlops1-blazingtext-endpoint',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        result_str = response['Body'].read().decode('utf-8')
        result_json = json.loads(result_str)
        
        label = result_json[0]['label'][0].replace('__label__', '')
        confidence = float(result_json[0]['prob'][0])

        return label, confidence

    except Exception as e:
        print(f"SageMaker BlazingText Hatası: {str(e)}. (N/A, 0.0) dönülüyor.")
        return "N/A", 0.0

def insert_dynamodb(comment_id, original_comment, btext_label, btext_confidence, kumru_label, kumru_confidence, final_label, gemini_called):
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table = dynamodb.Table('youtube_comment_analysis')
    
    table.put_item(
        Item={
            'comment_id': comment_id,
            'message': original_comment,
            'blazingtext_label': btext_label,              
            'blazingtext_confidence': Decimal(str(btext_confidence)),
            'kumru_label': kumru_label,
            'kumru_confidence': Decimal(str(kumru_confidence)),
            'final_label': final_label,
            'gemini_called': gemini_called,
            'timestamp': int(time.time())
        }
    )

def lambda_handler(event, context):
    try:
        kumru_label = "N/A"
        kumru_confidence = 0.0
        gemini_called = False
        source_model = "N/A" 

        body = json.loads(event.get("body", "{}"))
        user_message = body.get("message", "")
        
        if not user_message:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Mesaj boş olamaz"}, ensure_ascii=False)
            }
        
        blazingtext_label, blazingtext_prob = get_blazingtext_data(user_message)
        comment_id = str(uuid.uuid4())
        if blazingtext_prob >= SAGEMAKER_CONFIDENCE_THRESHOLD:
            final_label = blazingtext_label
            source_model = "BlazingText"

        elif blazingtext_prob < SAGEMAKER_CONFIDENCE_THRESHOLD:            
            try:
                kumru_data = json.loads(get_kumru_message(user_message))
                kumru_label = transliterate(kumru_data.get("predicted_label", "notr"))
                kumru_confidence = float(kumru_data.get("confidence", 0.0))
            except Exception as e:
                print(f"Kumru Modeli Hatası: {str(e)}. Gemini'ye geçiliyor.")
                kumru_label = "ERROR"
            
            if kumru_confidence >= CONFIDENCE_THRESHOLD:
                final_label = kumru_label
                source_model = "Kumru"
                gemini_called = False
                
            else:                
                try:
                    final_label = get_gemini_label(user_message)
                    source_model = "Gemini"
                    gemini_called = True
                except Exception as e:
                    print(f"Gemini API Hatası: {str(e)}. Son Etiket: notr.")
                    final_label = "notr"
                    source_model = "Gemini_Failed"
                    gemini_called = True
        
        insert_dynamodb(
            comment_id, 
            user_message, 
            blazingtext_label, 
            blazingtext_prob, 
            kumru_label, 
            kumru_confidence, 
            final_label, 
            gemini_called
        )
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "comment_id": comment_id,
                "final_label": final_label,
                "source": source_model,
                "blazingtext_label": blazingtext_label,
                "blazingtext_confidence": blazingtext_prob,
                "kumru_label": kumru_label,
                "kumru_confidence": kumru_confidence,
                "gemini_called": gemini_called
            }, ensure_ascii=False)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Hata: {str(e)}"}, ensure_ascii=False)
        }