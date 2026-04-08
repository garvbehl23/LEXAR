# Services

The Fonada SDK provides several services for managing your AI platform.

## Available Services

| Service | Description | Access |
|---------|-------------|--------|
| **Agents** | Agent CRUD, chat, API keys | API Key + JWT |
| **Campaigns** | Campaign management | JWT required |
| **Credits** | Balance, billing, payments | JWT required |
| **Knowledge Base** | Documents, RAG, search | JWT required |
| **Users** | User management | JWT required |
| **Organizations** | Org management | JWT required |
| **MCP Tools** | External integrations | JWT required |

## Agents Service

Manage AI agents and chat interactions.

### List Agents

```python
agents = client.agents.list()
for agent in agents:
    print(f"{agent.name}: {agent.llm_provider}/{agent.llm_model}")
```

### Get Agent

```python
agent = client.agents.get("agent-uuid")
print(f"Name: {agent.name}")
print(f"Description: {agent.description}")
print(f"Channel: {agent.channel}")
```

### Create Agent

```python
from fonada.models import Channel, LLMProvider, AgentMode

agent = client.agents.create(
    name="Support Bot",
    channel=Channel.WHATSAPP,
    llm_provider=LLMProvider.OPENAI,
    llm_model="gpt-4",
    system_prompt="You are a helpful assistant.",
    languages=["English"],
    agent_mode=AgentMode.CHAT
)
```

### Update Agent

```python
agent = client.agents.update(
    agent_id="agent-uuid",
    name="Updated Bot Name",
    system_prompt="Updated instructions..."
)
```

### Delete Agent

```python
client.agents.delete("agent-uuid")
```

### Chat with Agent

```python
response = client.agents.chat(
    bot_id="agent-uuid",
    message="Hello, I need help!",
    session_id="optional-session-id"
)
print(f"Reply: {response.reply}")
print(f"Session: {response.session_id}")
```

### Generate Agent API Key

```python
key = client.agents.generate_api_key("agent-uuid")
print(f"API Key: {key.api_key}")
```

[Full Agents Documentation →](services/agents.md)

## Campaigns Service

Manage marketing and outreach campaigns.

### Create WhatsApp Campaign

```python
campaign = client.campaigns.create_whatsapp(
    campaign_name="March Promo",
    template_id="template-uuid",
    account_id="account-uuid",
    csv_file=open("contacts.csv", "rb"),
    campaign_type="marketing"
)
```

### Campaign Control

```python
# Pause campaign
client.campaigns.pause("campaign-uuid")

# Resume campaign
client.campaigns.resume("campaign-uuid")

# Cancel campaign
client.campaigns.cancel("campaign-uuid")
```

### Get Campaign Status

```python
status = client.campaigns.get_status("campaign-uuid")
print(f"Status: {status.status}")
print(f"Progress: {status.progress}%")
```

[Full Campaigns Documentation →](services/campaigns.md)

## Credits Service

Manage credits, billing, and payments.

### Check Balance

```python
balance = client.credits.get_balance()
print(f"Balance: {balance.balance}")
```

### Get Usage Analytics

```python
usage = client.credits.get_usage_analytics(month="2026-03")
print(f"Total spent: {usage.total_spent}")
```

### Get Transactions

```python
transactions = client.credits.get_transactions(limit=50)
for tx in transactions:
    print(f"{tx.created_at}: {tx.amount} ({tx.service_key})")
```

### Initiate Payment

```python
payment = client.credits.initiate_payment(
    billing_cycle="one_time",
    payment_type="topup",
    customer_name="John Doe",
    customer_email="john@example.com",
    customer_phone="9876543210",
    success_url="https://app/success",
    failure_url="https://app/failure",
    topup_amount=1000
)
```

[Full Credits Documentation →](services/credits.md)

## Knowledge Base Service

Manage documents and RAG (Retrieval-Augmented Generation).

### Add Text Content

```python
client.knowledge_base.add_text(
    agent_id="agent-uuid",
    text="Our business hours are 9 AM to 6 PM, Monday to Friday."
)
```

### Add Webpage

```python
client.knowledge_base.add_webpage(
    agent_id="agent-uuid",
    url="https://example.com/faq"
)
```

### Process Document

```python
result = client.knowledge_base.process_document(
    agent_id="agent-uuid",
    file_path="manual.pdf"
)
```

### Search Knowledge Base

```python
results = client.knowledge_base.search(
    agent_id="agent-uuid",
    query="What are the return policies?"
)
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
```

### Crawl Website

```python
result = client.knowledge_base.crawl_website(
    agent_id="agent-uuid",
    url="https://example.com",
    max_pages=50
)
```

[Full Knowledge Base Documentation →](services/knowledge-base.md)

## Users Service

Manage platform users.

### Create User

```python
user = client.users.create(
    email="user@example.com",
    full_name="John Doe",
    role="user"
)
```

### List Users

```python
users = client.users.list()
```

### Update User

```python
client.users.update(
    user_id="user-uuid",
    role="admin",
    is_active=True
)
```

### Generate API Key

```python
key = client.users.generate_api_key()
print(f"New API Key: {key.api_key}")
```

[Full Users Documentation →](services/users.md)

## Organizations Service

Manage organizations and teams.

### Create Organization

```python
org = client.organizations.create(
    name="acme",
    display_name="Acme Corporation",
    org_type="enterprise",
    billing_type="prepaid"
)
```

### Allocate Credits

```python
client.organizations.allocate_credits(
    organization_id="org-uuid",
    amount=5000,
    reason="Monthly allocation"
)
```

### Add User to Organization

```python
client.organizations.add_user(
    organization_id="org-uuid",
    user_id="user-uuid",
    role_key="admin"
)
```

## MCP Tools Service

Manage external tool integrations.

### List Available Tools

```python
tools = client.mcp_tools.list_tools()
for tool in tools:
    print(f"Tool: {tool['name']}")
```

### Initiate OAuth

```python
auth = client.mcp_tools.initiate_oauth(
    platform_name="salesforce",
    connection_label="My Salesforce"
)
print(f"Auth URL: {auth['auth_url']}")
```

### List Connections

```python
connections = client.mcp_tools.list_connections()
```
