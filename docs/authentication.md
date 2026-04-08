# Authentication

The Fonada SDK supports multiple authentication methods to suit different use cases.

## Authentication Methods

| Method | Token Type | Access Level | Best For |
|--------|------------|--------------|----------|
| **API Key** | `fsk_*` | Read-only (agents) | Server-side scripts |
| **JWT Token** | `eyJ*` | Full access | User sessions |
| **Agent Key** | Agent-specific | Agent endpoints | Web widgets |
| **Service Role** | `eyJ*` (service_role) | Admin access | Backend systems |

## API Key Authentication

The Fonada API Key (`fsk_` prefix) provides secure access to the platform.

### Getting Your API Key

1. Log in to [prod.fonada.ai](https://prod.fonada.ai)
2. Go to Settings → Developer → API Keys
3. Generate a new API key
4. Copy the key (starts with `fsk_`)

### Usage

```python
from fonada import FonadaClient

client = FonadaClient(
    api_key="fsk_your_api_key",
    project_url="https://your-project.supabase.co"
)
```

### Environment Variable

```bash
export FONADA_API_KEY=fsk_your_api_key
export FONADA_PROJECT_URL=https://your-project.supabase.co
```

```python
from fonada import FonadaClient

# Auto-loads from environment
client = FonadaClient()
```

### Access Level

With API Key, you can:
- ✅ List agents
- ✅ Get agent details
- ❌ Chat with agents (requires JWT)
- ❌ Access credits/billing (requires JWT)
- ❌ Send messages (requires JWT)

## JWT Token Authentication

JWT tokens provide full access to all platform features.

### Getting Your JWT Token

1. Log in to [prod.fonada.ai](https://prod.fonada.ai)
2. Open Developer Tools (F12)
3. Go to Application → Local Storage
4. Find `sb-{project-ref}-auth-token`
5. Copy the `access_token` value

### Usage

```python
from fonada import FonadaClient

client = FonadaClient(
    jwt_token="eyJhbGciOiJIUzI1NiIs...",
    project_url="https://your-project.supabase.co",
    supabase_anon_key="eyJhbGciOiJIUzI1NiIs..."  # Required for Edge Functions
)
```

### Token Expiration

JWT tokens expire after approximately 1 hour. For long-running applications, implement token refresh or use API keys where possible.

## Agent Key Authentication

Agent keys are specific to individual agents and are used for web widget integrations.

### Generating Agent Key

```python
# Using an authenticated client
key_response = client.agents.generate_api_key(agent_id="your-agent-uuid")
print(f"Agent Key: {key_response.api_key}")
```

### Usage

```python
from fonada import FonadaClient

client = FonadaClient(
    agent_key="agent_specific_key",
    project_url="https://your-project.supabase.co"
)

# Use for web widget chat
response = client.agents.chat_web_widget(
    bot_id="agent-uuid",
    message="Hello!",
    api_key="agent_specific_key"
)
```

## Supabase Anon Key

The Supabase anonymous key is required for Edge Function access.

### Finding Your Anon Key

1. Go to your Supabase Dashboard
2. Navigate to Settings → API
3. Copy the `anon` public key

### Usage

```python
client = FonadaClient(
    api_key="fsk_xxx",
    project_url="https://xxx.supabase.co",
    supabase_anon_key="eyJhbGciOiJIUzI1NiIs..."
)
```

## Complete Configuration Example

```python
from fonada import FonadaClient
import os

# Full configuration
client = FonadaClient(
    # Authentication (one of these required)
    api_key=os.getenv("FONADA_API_KEY"),
    jwt_token=os.getenv("FONADA_JWT_TOKEN"),
    
    # Project URL (required)
    project_url=os.getenv("FONADA_PROJECT_URL"),
    
    # Optional but recommended
    supabase_anon_key=os.getenv("FONADA_ANON_KEY"),
    
    # User context (for some operations)
    user_id=os.getenv("FONADA_USER_ID"),
    
    # Request settings
    timeout=30.0,
    max_retries=3
)
```

## Authentication Priority

When multiple authentication methods are provided, the SDK uses this priority:

1. **JWT Token** (highest priority - full access)
2. **API Key** (read-only access)
3. **Agent Key** (agent-specific access)

## Security Best Practices

1. **Never commit credentials** - Use environment variables or `.env` files
2. **Keep `.env` in `.gitignore`** - Already configured in this SDK
3. **Rotate keys periodically** - Generate new keys and revoke old ones
4. **Use minimal permissions** - Use API key for read operations
5. **Protect service role keys** - Never expose in client-side code

## Troubleshooting

### Error: "Invalid Token or Protected Header formatting"

This usually means:
- The token format is incorrect
- You're using anon key instead of JWT
- The endpoint requires JWT but you're using API key

### Error: "User authentication required"

This endpoint requires:
- A valid JWT token from user login, OR
- User context (user_id) in the request

### Error: "401 Unauthorized"

Check:
- API key is correct and not expired
- Project URL matches your Supabase project
- Required headers are being sent
