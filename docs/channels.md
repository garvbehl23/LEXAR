# Channels

The Fonada SDK supports multiple communication channels for deploying AI agents.

## Supported Channels

| Channel | Description | Status |
|---------|-------------|--------|
| **WhatsApp** | WhatsApp Business API | ✅ Full support |
| **RCS** | Rich Communication Services | ✅ Full support |
| **Voice** | Telephony and LiveKit | ✅ Full support |
| **Instagram** | Instagram Direct Messages | ✅ Basic support |
| **Messenger** | Facebook Messenger | ✅ Basic support |
| **LinkedIn** | LinkedIn Messaging | ✅ Basic support |

## Channel Overview

### WhatsApp

The most feature-rich channel with support for:
- Text messages
- Media messages (image, video, document, audio)
- Template messages
- Interactive messages (buttons, lists)
- Contact management
- Session management

```python
# Send text message
client.whatsapp.send_text(
    account_id="uuid",
    to="+919876543210",
    text="Hello! How can I help you?"
)

# Send template message
client.whatsapp.send_template(
    phone="+919876543210",
    template_name="order_confirmation",
    template_language="en",
    variables={"order_id": "12345", "amount": "500"}
)
```

[Full WhatsApp Documentation →](channels/whatsapp.md)

### RCS

Rich Communication Services for enhanced messaging:
- Rich media messages
- Suggested actions
- Carousels
- Template support

```python
# Send RCS message
client.rcs.send_message(
    phone_number="+919876543210",
    template_key="greeting",
    template_value="Hello, welcome!"
)
```

[Full RCS Documentation →](channels/rcs.md)

### Voice/Telephony

Voice calling with AI agents:
- LiveKit integration for real-time voice
- Call routing
- IVR systems
- Recording management

```python
# Get LiveKit token
token = client.voice.get_livekit_token(
    room_name="support-room",
    participant_name="customer-123"
)
```

[Full Voice Documentation →](channels/voice.md)

### Social Media Channels

Instagram, Messenger, and LinkedIn support:
- OAuth authentication
- Direct messaging
- Conversation management

```python
# Instagram
client.instagram.send_message(
    agent_id="uuid",
    recipient_id="instagram_user_id",
    text="Hello!"
)

# Messenger
client.messenger.send_message(
    agent_id="uuid",
    recipient_id="messenger_user_id",
    message="Hello!"
)

# LinkedIn
client.linkedin.send_message(
    agent_id="uuid",
    recipient_id="linkedin_user_id",
    message="Hello!"
)
```

[Full Social Media Documentation →](channels/social-media.md)

## Channel Access

All channels are accessed through the main client:

```python
from fonada import FonadaClient

client = FonadaClient(...)

# Access channels
client.whatsapp    # WhatsApp operations
client.rcs         # RCS operations
client.voice       # Voice/Telephony operations
client.instagram   # Instagram operations
client.messenger   # Messenger operations
client.linkedin    # LinkedIn operations
```

## Agent Channel Configuration

When creating an agent, specify the target channel:

```python
from fonada.models import Channel

# WhatsApp agent
agent = client.agents.create(
    name="WhatsApp Bot",
    channel=Channel.WHATSAPP,
    llm_provider="openai",
    llm_model="gpt-4"
)

# Voice agent
agent = client.agents.create(
    name="Voice Bot",
    channel=Channel.TELEPHONY,
    llm_provider="openai",
    llm_model="gpt-4",
    tts_provider="elevenlabs",
    asr_provider="deepgram"
)
```

## Available Channel Types

```python
from fonada.models import Channel

Channel.WHATSAPP     # WhatsApp Business
Channel.RCS          # Rich Communication Services
Channel.TELEPHONY    # Voice calls
Channel.INSTAGRAM    # Instagram DM
Channel.MESSENGER    # Facebook Messenger
Channel.LINKEDIN     # LinkedIn messages
Channel.WEB_WIDGET   # Web chat widget
Channel.CHAT_BOT     # Generic chat
```
