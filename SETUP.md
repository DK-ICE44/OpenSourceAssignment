# HR/IT Copilot - Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env` and update with your credentials:

```bash
cp .env.example .env
```

Then edit `.env` with your API keys and settings.

### 3. Get API Keys

#### Google Gemini API (Required for chat)
1. Go to https://ai.google.dev
2. Click "Get API Key"
3. Create a new project or select an existing one
4. Copy your API key
5. Add to `.env`: `GEMINI_API_KEY=your-key-here`

#### Groq API (Required for intent classification)
1. Go to https://console.groq.com
2. Sign up or log in
3. Create an API key
4. Copy your API key  
5. Add to `.env`: `GROQ_API_KEY=your-key-here`

#### Change Secret Key (Important for Production)
Generate a new secret key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Add to `.env`: `SECRET_KEY=your-generated-key`

### 4. Initialize Database and Seed Sample Data

```bash
python -m app.seed
```

This creates:
- 5 sample users (EMP001, EMP002, HR001, IT001, ADM001)
- Leave balances for 2026
- Holiday calendar

### 5. Run the Application

```bash
uvicorn app.main:app --reload
```

The app will start at `http://localhost:8000`

## Testing the Chat Endpoint

### 1. Login to Get Auth Token

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=alice@company.com&password=password123"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Use the Token to Chat

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the work from home policy?"}'
```

### 3. Or Use the Interactive Docs

1. Go to http://localhost:8000/docs (Swagger UI)
2. Click "Authorize" and paste your token
3. Find the POST /chat endpoint and try it

## Sample Users

| Employee ID | Name | Email | Password | Role |
|---|---|---|---|---|
| EMP001 | Alice Johnson | alice@company.com | password123 | Employee |
| EMP002 | Bob Smith | bob@company.com | password123 | Manager |
| HR001 | Carol White | carol@company.com | password123 | HR Team |
| IT001 | Dave Brown | dave@company.com | password123 | IT Team |
| ADM001 | Eve Admin | eve@company.com | password123 | Admin |

## Troubleshooting

### 500 Internal Server Error on /chat

**Cause**: Missing or invalid API keys in `.env`

**Solution**: 
1. Make sure `.env` file exists (not just `.env.example`)
2. Add valid `GEMINI_API_KEY` and `GROQ_API_KEY`
3. Restart the server: `uvicorn app.main:app --reload`

### 401 Unauthorized Error

**Cause**: Missing or invalid authentication token

**Solution**:
1. Login first to get a token: `POST /auth/login`
2. Pass the token in the Authorization header: `Authorization: Bearer <your-token>`

### 422 Unprocessable Entity Error

**Cause**: Invalid request body

**Solution**: 
- Ensure the request body is valid JSON: `{"message": "your message here"}`
- Don't use GET, only POST requests

## API Endpoints

### Authentication
- `POST /auth/register` - Register a new user
- `POST /auth/login` - Login and get access token
- `POST /auth/refresh` - Refresh the access token

### Chat
- `POST /chat` - Send a message (requires auth token)
- `DELETE /chat/memory` - Clear conversation history

### HR Management
- `GET /hr/leave/balance` - Check leave balance
- `POST /hr/leave/apply` - Apply for leave
- `GET /hr/leave/my-requests` - View your leave requests
- `POST /hr/leave/approve` - Approve/reject leave (manager only)

### IT Support
- `POST /it/tickets/create` - Create a support ticket
- `GET /it/tickets/list` - List your tickets
- `POST /it/assets/request` - Request IT equipment
- `GET /it/assets/list` - View your asset requests

## Features

### Multi-Agent System
The app uses LangGraph with multiple specialized agents:
- **Router Agent**: Classifies user intent
- **HR RAG Agent**: Answers policy questions using retrieval
- **HR Leave Agent**: Manages leave requests
- **IT Support Agent**: Handles IT tickets and assets
- **General Agent**: Handles greetings and unclear queries

### Memory Management
- Conversation history is persisted per employee
- Each employee gets their own thread_id based on employee_id
- Memory is stored in SQLite with LangGraph's SqliteSaver

### Multi-turn Conversations
- The system remembers previous messages in the conversation
- Full chat history is maintained across sessions
- Can clear memory with `DELETE /chat/memory`

## Database

The app uses SQLite by default (`db/copilot.db`). To use a different database, update the `DATABASE_URL` in `.env`:

```
# PostgreSQL example:
DATABASE_URL=postgresql://user:password@localhost/copilot_db

# MySQL example:
DATABASE_URL=mysql://user:password@localhost/copilot_db
```

## Development

### Run Tests
```bash
pytest tests/
```

### Run with Debug Logging
```bash
DEBUG=true uvicorn app.main:app --reload
```

### Enable LangSmith Tracing
1. Get an API key from https://smith.langchain.com
2. Add to `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-key-here
   ```

## License

This project is proprietary.
