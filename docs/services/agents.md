# Agents Service

Complete guide for managing AI agents and chat interactions.

## Overview

The Agents service provides:
- Full CRUD operations for agents
- Chat functionality (text and voice)
- API key management for agents
- Agent configuration retrieval

## Agent Operations

### List Agents

```python
agents = client.agents.list()
for agent in agents:
    print(f"Name: {agent.name}")
    print(f"ID: {agent.id}")
    print(f"Provider: {agent.llm_provider}")
    print(f"Model: {agent.llm_model}")
    print(f"Channel: {agent.channel}")
    print(f"Mode: {agent.agent_mode}")
    print()
```

### Get Agent Details

```python
agent = client.agents.get("agent-uuid")

print(f"Name: {agent.name}")
print(f"Description: {agent.description}")
print(f"System Prompt: {agent.system_prompt}")
print(f"Languages: {agent.languages}")
print(f"Is Active: {agent.is_active}")
print(f"Is Deployed: {agent.is_deployed}")

# Access extra fields from server
if hasattr(agent, 'model_extra'):
    print(f"Extra fields: {agent.model_extra.keys()}")
```

### Create Agent

```python
from fonada.models import Channel, LLMProvider, AgentMode, AgentType

agent = client.agents.create(
    # Required
    name="Customer Support Bot",
    channel=Channel.WHATSAPP,
    
    # LLM Configuration
    llm_provider=LLMProvider.OPENAI,
    llm_model="gpt-4",
    system_prompt="""You are a helpful customer support assistant.
    Be polite, concise, and helpful.
    If you don't know something, say so.""",
    
    # Agent behavior
    agent_mode=AgentMode.CHAT,
    agent_type=AgentType.INBOUND,
    
    # Languages
    languages=["English", "Hindi"],
    
    # Optional: Description
    description="Handles customer inquiries for our e-commerce platform",
    
    # Optional: Voice settings (for telephony)
    tts_provider="elevenlabs",
    asr_provider="deepgram",
    voice="rachel"
)

print(f"Created agent: {agent.id}")
```

### Update Agent

```python
agent = client.agents.update(
    agent_id="agent-uuid",
    name="Updated Bot Name",
    system_prompt="Updated system instructions...",
    llm_model="gpt-4-turbo"
)
```

### Delete Agent

```python
result = client.agents.delete("agent-uuid")
print(f"Deleted: {result.get('success')}")
```

## Chat Operations

### Basic Chat

```python
response = client.agents.chat(
    bot_id="agent-uuid",
    message="Hello, I need help with my order"
)

print(f"Reply: {response.reply}")
print(f"Session ID: {response.session_id}")
print(f"Provider: {response.llm_provider}")
print(f"Model: {response.llm_model}")
```

### Chat with Session

```python
# First message - creates session
response1 = client.agents.chat(
    bot_id="agent-uuid",
    message="What's the status of order #12345?"
)
session_id = response1.session_id

# Continue conversation
response2 = client.agents.chat(
    bot_id="agent-uuid",
    message="Can you expedite the delivery?",
    session_id=session_id
)

# End session
response3 = client.agents.chat(
    bot_id="agent-uuid",
    message="Thanks, that's all!",
    session_id=session_id,
    end_session=True
)
```

### Reset Session

```python
response = client.agents.chat(
    bot_id="agent-uuid",
    message="",
    action="reset_session",
    session_id=session_id
)
```

### Multimodal Chat (with images)

```python
response = client.agents.chat(
    bot_id="agent-uuid",
    message="What product is this?",
    images=["https://example.com/product.jpg"]
)
```

### Voice Chat

```python
response = client.agents.chat_voice(
    agent_id="agent-uuid",
    transcript="Hello, I need help",
    user_id="user-uuid",
    session_id="session-uuid",
    context={"call_id": "call-123"}
)
```

### Web Widget Chat

```python
response = client.agents.chat_web_widget(
    bot_id="agent-uuid",
    message="Hello!",
    api_key="agent-api-key",
    session_id="optional-session-id"
)
```

## API Key Management

### Generate API Key

```python
key_response = client.agents.generate_api_key("agent-uuid")

print(f"Success: {key_response.success}")
print(f"Agent ID: {key_response.agent_id}")
print(f"API Key: {key_response.api_key}")
print(f"Webhook URLs: {key_response.webhook_urls}")
```

### Get Existing Key

```python
key_response = client.agents.get_api_key("agent-uuid")
print(f"API Key: {key_response.api_key}")
```

### Revoke Key

```python
key_response = client.agents.revoke_api_key("agent-uuid")
print(f"Revoked: {key_response.success}")
```

## Agent Configuration

### Get Public Config

```python
# By agent ID
config = client.agents.get_config(agent_id="agent-uuid")

# By phone number
config = client.agents.get_config(phone_number="+919876543210")

# By WhatsApp number
config = client.agents.get_config(whatsapp_number="+919876543210")

# By campaign ID
config = client.agents.get_config(fonada_campaign_id="campaign-uuid")
```

### Invalidate Cache

```python
result = client.agents.invalidate_cache("agent-uuid")
print(f"Cache invalidated: {result.get('success')}")
```

## Agent Models

### Channel Options

```python
from fonada.models import Channel

Channel.WHATSAPP     # WhatsApp Business
Channel.RCS          # Rich Communication Services
Channel.TELEPHONY    # Voice calls
Channel.INSTAGRAM    # Instagram DM
Channel.MESSENGER    # Facebook Messenger
Channel.LINKEDIN     # LinkedIn
Channel.WEB_WIDGET   # Embedded chat
Channel.CHAT_BOT     # Generic chat
```

### LLM Providers

```python
from fonada.models import LLMProvider

LLMProvider.OPENAI       # OpenAI GPT models
LLMProvider.ANTHROPIC    # Claude models
LLMProvider.GOOGLE       # Google AI models
LLMProvider.GROQ         # Groq models
LLMProvider.AZURE        # Azure OpenAI
LLMProvider.DEEPSEEK     # DeepSeek models
LLMProvider.GEMINI       # Google Gemini
```

### Agent Modes

```python
from fonada.models import AgentMode

AgentMode.CHAT    # Conversational AI
AgentMode.FLOW    # Flow-based (linked to flows)
AgentMode.LLM     # Pure LLM responses
```

## Best Practices

1. **Use descriptive names** for agents
2. **Write clear system prompts** with specific instructions
3. **Set appropriate languages** for your audience
4. **Use sessions** for multi-turn conversations
5. **Handle errors** gracefully in chat
6. **Secure API keys** - never expose in client code
7. **Monitor agent performance** through analytics
