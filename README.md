# Autonomous-MLOps-Platform-for-Real-Time-Content-Analysis-

Engineered a fully autonomous, event-driven MLOps pipeline on AWS that automatically retrains, evaluates, and deploys a text classification model with zero downtime, triggered by data volume thresholds in DynamoDB Streams.

Designed and implemented a tiered, hybrid AI inference engine to dynamically balance cost and accuracy, prioritizing a custom-trained SageMaker model, escalating to a specialized NLP API (Kumru), and finally to a large language model (Gemini) for the most complex cases.

Architected the entire system on a serverless stack (Lambda, Step Functions, API Gateway, DynamoDB) to ensure high availability and scalability while minimizing operational overhead and cost.

Developed a real-time observability dashboard backend that calculates and exposes key performance indicators (KPIs) such as model mismatch rates, cost-per-inference, and source model distribution, enabling data-driven system optimization.

Automated the model deployment process using AWS EventBridge to listen for SageMaker training completion events, triggering a Lambda function that orchestrates a Blue/Green deployment by creating new model versions and updating the live endpoints seamlessly
