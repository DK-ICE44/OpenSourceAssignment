#!/usr/bin/env python
"""Test the chat API endpoint"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Step 1: Login
print("Step 1: Logging in...")
login_data = {
    "employee_id": "EMP001",
    "password": "password123"
}

login_response = requests.post(
    f"{BASE_URL}/auth/login",
    json=login_data,
    headers={"Content-Type": "application/json"}
)

print(f"Login Status: {login_response.status_code}")
print(f"Login Response: {login_response.text}")

if login_response.status_code != 200:
    print("❌ Login failed!")
    exit(1)

token = login_response.json()["access_token"]
print(f"✅ Login successful! Token: {token[:20]}...")

# Step 2: Test chat endpoint
print("\nStep 2: Testing chat endpoint...")
chat_data = {
    "message": "What is the work from home policy?"
}

chat_response = requests.post(
    f"{BASE_URL}/chat",
    json=chat_data,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
)

print(f"Chat Status: {chat_response.status_code}")
print(f"Chat Response:")
print(json.dumps(chat_response.json(), indent=2))

if chat_response.status_code == 200:
    print("\n✅ Chat endpoint working!")
else:
    print(f"\n❌ Chat endpoint failed with status {chat_response.status_code}")
