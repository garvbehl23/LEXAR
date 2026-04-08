# Getting Started

This guide will help you get started with the Fonada AI Python SDK.

## Prerequisites

- Python 3.8 or higher
- A Fonada AI account ([sign up](https://prod.fonada.ai))
- Your API credentials

## Installation

### From PyPI (when published)

```bash
pip install fonada-sdk
```

### From Source

```bash
git clone https://github.com/Shivtel-pvt-Ltd/sdk.git
cd sdk
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Required
FONADA_API_KEY=fsk_your_api_key_here
FONADA_PROJECT_URL=https://your-project.supabase.co

# Optional - for Edge Function access
FONADA_ANON_KEY=your_supabase_anon_key

# Optional - for full access (chat, credits, etc.)
FONADA_JWT_TOKEN=your_jwt_token

# Optional - user context
FONADA_USER_ID=your_user_uuid
```

### Initialize the Client

```python
from fonada import FonadaClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Option 1: Auto-load from environment
client = FonadaClient()

# Option 2: Explicit configuration
client = FonadaClient(
    api_key="fsk_xxx",
    project_url="https://xxx.supabase.co",
    timeout=30.0,
    max_retries=3
)
```

## Basic Usage

### List Agents

```python
from fonada import FonadaClient
from dotenv import load_dotenv

load_dotenv()
client = FonadaClient()

# Get all agents
agents = client.agents.list()
for agent in agents:
    print(f"Name: {agent.name}")
    print(f"  ID: {agent.id}")
    print(f"  Provider: {agent.llm_provider}")
    print(f"  Model: {agent.llm_model}")
    print(f"  Channel: {agent.channel}")
    print()
```

### Get Agent Details

```python
agent_id = "your-agent-uuid"
agent = client.agents.get(agent_id)

print(f"Agent: {agent.name}")
print(f"Description: {agent.description}")
print(f"System Prompt: {agent.system_prompt}")
print(f"Languages: {agent.languages}")
```

### Create an Agent

```python
from fonada.models import Channel, LLMProvider, AgentMode

agent = client.agents.create(
    name="Customer Support Bot",
    channel=Channel.WHATSAPP,
    llm_provider=LLMProvider.OPENAI,
    llm_model="gpt-4",
    system_prompt="You are a helpful customer support assistant.",
    languages=["English", "Hindi"],
    agent_mode=AgentMode.CHAT
)

print(f"Created agent: {agent.id}")
```

## Project Structure

```
fonada-sdk/
├── src/fonada/
│   ├── __init__.py          # Package exports
│   ├── client.py             # Main client class
│   ├── auth.py               # Authentication handlers
│   ├── http_client.py        # HTTP request handling
│   ├── exceptions.py         # Custom exceptions
│   ├── channels/             # Channel implementations
│   │   ├── whatsapp.py
│   │   ├── rcs.py
│   │   ├── voice.py
│   │   ├── instagram.py
│   │   ├── messenger.py
│   │   └── linkedin.py
│   ├── services/             # Service implementations
│   │   ├── agents.py
│   │   ├── campaigns.py
│   │   ├── credits.py
│   │   ├── knowledge_base.py
│   │   ├── users.py
│   │   ├── organizations.py
│   │   └── mcp_tools.py
│   ├── models/               # Pydantic models
│   │   ├── agent.py
│   │   ├── campaign.py
│   │   ├── credits.py
│   │   ├── message.py
│   │   └── user.py
│   └── utils/                # Utility functions
│       └── validators.py
├── tests/                    # Unit tests
├── examples/                 # Example scripts
├── docs/                     # Documentation
└── pyproject.toml           # Project configuration
```

## Next Steps

- [Authentication Guide](authentication.md)
- [Working with Channels](channels.md)
- [Using Services](services.md)
- [Error Handling](error-handling.md)
