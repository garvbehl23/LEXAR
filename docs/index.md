# Fonada AI Python SDK Documentation

Welcome to the official documentation for the Fonada AI Python SDK.

## Overview

The Fonada AI Python SDK provides a simple and intuitive interface to interact with the Fonada AI Platform. Build and deploy AI agents across multiple channels including WhatsApp, Voice, RCS, Instagram, Messenger, and LinkedIn.

## Table of Contents

1. [Getting Started](getting-started.md)
2. [Authentication](authentication.md)
3. [Channels](channels.md)
   - [WhatsApp](channels/whatsapp.md)
   - [RCS](channels/rcs.md)
   - [Voice/Telephony](channels/voice.md)
   - [Social Media](channels/social-media.md)
4. [Services](services.md)
   - [Agents](services/agents.md)
   - [Campaigns](services/campaigns.md)
   - [Credits & Billing](services/credits.md)
   - [Knowledge Base](services/knowledge-base.md)
   - [Users & Organizations](services/users.md)
5. [Error Handling](error-handling.md)
6. [API Reference](api-reference.md)

## Quick Links

- [GitHub Repository](https://github.com/Shivtel-pvt-Ltd/sdk)
- [Fonada Platform](https://prod.fonada.ai)
- [API Documentation](../API_DOCUMENTATION.md)

## Installation

```bash
pip install fonada-sdk
```

Or install from source:

```bash
git clone https://github.com/Shivtel-pvt-Ltd/sdk.git
cd sdk
pip install -e .
```

## Quick Example

```python
from fonada import FonadaClient

# Initialize client
client = FonadaClient(
    api_key="fsk_your_api_key",
    project_url="https://your-project.supabase.co"
)

# List your agents
agents = client.agents.list()
for agent in agents:
    print(f"Agent: {agent.name} ({agent.llm_provider})")
```

## Support

- Email: support@fonada.ai
- Platform: https://prod.fonada.ai
