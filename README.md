# AI-Powered-Credit-Scoring-System

AI-Powered Credit Scoring: An advanced system will evaluate the borrower’s creditworthiness using AI. It will analyze financial documents (e.g., bank statements, tax returns), income history (e.g., monthly or annual income), and credit scores from bureaus. This will help determine the likelihood of loan approval.

Automated Loan Underwriting
AI-Based Risk Evaluation:: The AI will analyze a borrower's financial history, payment behavior (e.g., timely or late payments), and income stability. These factors will help to calculate the risk involved in lending to the borrower.
Loan Risk Categorization: Then the applied Loans will be automatically assigned risk levels—Low, Medium, or High—based on the AI’s analysis.

Feedback: The borrowers will get immediate results based on the AI analysis.
This will include:
    Whether they qualify for the loan or not.
    tentative loan terms, such as the maximum amount they can borrow and the repayment period.
    Estimated interest rates, helping borrowers understand costs upfront.
-------------------
To develop an AI-powered credit scoring system that evaluates a borrower’s creditworthiness, assesses loan risks, and automates loan underwriting, we can use Python and machine learning tools. The system would involve:

    Financial Data Processing: Analyzing bank statements, tax returns, and other financial documents.
    Income History Evaluation: Analyzing the borrower's income and financial stability.
    Credit Score Analysis: Using external APIs to fetch credit scores or relying on internal financial data.
    Risk Evaluation: Categorizing the risk level for a loan based on historical data and predictive models.

Key Steps Involved:

    Data Preprocessing: Extracting data from documents and structured sources.
    Feature Extraction: Identifying key features from financial documents (e.g., income, liabilities, payment history).
    Credit Scoring: Applying a machine learning model to calculate the credit score or likelihood of loan approval.
    Loan Risk Categorization: Classifying the loan into risk categories (low, medium, high).
    Automated Decision Feedback: Providing the borrower with the loan outcome.

Prerequisites:

    Install required libraries:

pip install pandas scikit-learn numpy openpyxl requests

Python Code Implementation:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import openai  # For document analysis (optional if using AI-based processing for documents)

# Sample loan application data (for demonstration purposes)
# You can use actual datasets (bank statements, tax returns, etc.)
loan_data = pd.DataFrame({
    'income': [50000, 70000, 80000, 100000, 45000, 60000],
    'debt': [10000, 20000, 15000, 30000, 12000, 10000],
    'credit_score': [650, 700, 750, 800, 620, 680],
    'payment_behavior': [1, 1, 1, 1, 0, 1],  # 1: Timely payments, 0: Late payments
    'loan_approved': [1, 1, 1, 1, 0, 1]  # 1: Approved, 0: Denied
})

# Define features and target variable
X = loan_data[['income', 'debt', 'credit_score', 'payment_behavior']]  # Features
y = loan_data['loan_approved']  # Target variable (loan approval)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to evaluate loan risk (based on AI model)
def evaluate_loan_risk(income, debt, credit_score, payment_behavior):
    input_data = np.array([[income, debt, credit_score, payment_behavior]])
    risk_prediction = model.predict(input_data)
    
    if risk_prediction == 1:
        return "Loan Approved"
    else:
        return "Loan Denied"

# Function to categorize loan risk (Low, Medium, High)
def categorize_loan_risk(income, debt, credit_score, payment_behavior):
    input_data = np.array([[income, debt, credit_score, payment_behavior]])
    risk_score = model.predict_proba(input_data)[0, 1]  # Probability of loan approval

    if risk_score > 0.8:
        return "Low Risk"
    elif 0.5 < risk_score <= 0.8:
        return "Medium Risk"
    else:
        return "High Risk"

# Function to estimate loan terms (based on risk category)
def estimate_loan_terms(risk_category):
    if risk_category == "Low Risk":
        return {"max_loan_amount": 50000, "interest_rate": 5, "repayment_period": 5}
    elif risk_category == "Medium Risk":
        return {"max_loan_amount": 30000, "interest_rate": 8, "repayment_period": 3}
    else:
        return {"max_loan_amount": 10000, "interest_rate": 12, "repayment_period": 1}

# Function to generate a loan feedback response
def loan_feedback(income, debt, credit_score, payment_behavior):
    loan_approval = evaluate_loan_risk(income, debt, credit_score, payment_behavior)
    risk_category = categorize_loan_risk(income, debt, credit_score, payment_behavior)
    loan_terms = estimate_loan_terms(risk_category)
    
    return {
        "loan_approval": loan_approval,
        "risk_category": risk_category,
        "loan_terms": loan_terms
    }

# Example usage of the system for a borrower
borrower_data = {
    'income': 70000,
    'debt': 15000,
    'credit_score': 690,
    'payment_behavior': 1  # Timely payments
}

# Get feedback for the borrower
feedback = loan_feedback(borrower_data['income'], borrower_data['debt'], borrower_data['credit_score'], borrower_data['payment_behavior'])
print("Loan Feedback:", feedback)

# Optional: Document analysis for extracting financial data (bank statements, tax returns)
def analyze_document(file_path):
    # Placeholder for AI-based document analysis (e.g., using OpenAI API for extracting financial data)
    with open(file_path, 'r') as file:
        document_text = file.read()

    # Use OpenAI to process and extract structured financial data (bank statements, income, etc.)
    openai.api_key = 'your-openai-api-key'  # Set your OpenAI API key

    # For demonstration, we can prompt OpenAI to extract income and debts from the document
    prompt = f"Extract financial data (income, debts, etc.) from the following document: {document_text}"
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=150)
    
    extracted_data = response.choices[0].text.strip()
    return extracted_data

# Example usage for analyzing financial documents (bank statements, tax returns, etc.)
# document_data = analyze_document('path_to_your_financial_document.txt')
# print("Extracted Financial Data:", document_data)

Key Steps:

    Data Preprocessing: Here, we use a sample loan_data dataframe. For real-world applications, you would extract data from sources like bank statements, tax returns, etc.

    Model Training:
        We train a Random Forest Classifier to predict whether the loan will be approved or not based on factors like income, debt, credit score, and payment behavior.

    Loan Risk Evaluation:
        Risk categorization is done by predicting the probability of loan approval and assigning a risk level (Low, Medium, or High).

    Feedback Generation:
        Based on the AI analysis, the borrower will get feedback on their loan approval status, the risk level, the maximum loan amount, interest rates, and repayment terms.

    Document Analysis (Optional):
        OpenAI API or similar AI tools can be used for extracting financial data from documents like bank statements or tax returns.

Next Steps:

    Data Integration:
        Integrate with APIs that provide credit scores from bureaus or retrieve financial documents from various sources.
    AI Optimization:
        Experiment with more advanced AI models for better predictions. You can also use NLP-based models for analyzing unstructured data (e.g., contracts, financial statements).
    UI/UX:
        Develop a user-friendly front end where borrowers can input their details and receive instant feedback.
    Compliance & Security:
        Ensure that the system complies with financial regulations and securely handles sensitive borrower data.

This approach uses machine learning for risk analysis and provides transparent feedback to borrowers, which enhances their understanding of the loan process and terms.
