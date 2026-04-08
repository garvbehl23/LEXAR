# API Reference

Complete reference for all SDK classes, methods, and models.

## Client

### FonadaClient

Main client for interacting with the Fonada API.

```python
class FonadaClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        agent_key: Optional[str] = None,
        project_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None
```

**Parameters:**
- `api_key`: Fonada API key (`fsk_` prefix)
- `jwt_token`: JWT token for full access
- `agent_key`: Agent-specific API key
- `project_url`: Supabase project URL
- `supabase_anon_key`: Supabase anonymous key
- `user_id`: User ID for user context
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum retry attempts

**Properties:**
- `whatsapp`: WhatsAppChannel
- `rcs`: RCSChannel
- `voice`: VoiceChannel
- `instagram`: InstagramChannel
- `messenger`: MessengerChannel
- `linkedin`: LinkedInChannel
- `agents`: AgentsService
- `campaigns`: CampaignsService
- `credits`: CreditsService
- `knowledge_base`: KnowledgeBaseService
- `users`: UsersService
- `organizations`: OrganizationsService
- `mcp_tools`: MCPToolsService

---

## Channels

### WhatsAppChannel

```python
# Send text message
send_text(account_id: str, to: str, text: str) -> Dict

# Send image
send_image(account_id: str, to: str, url: str, caption: str = None) -> Dict

# Send video
send_video(account_id: str, to: str, url: str, caption: str = None) -> Dict

# Send document
send_document(account_id: str, to: str, url: str, filename: str, caption: str = None) -> Dict

# Send audio
send_audio(account_id: str, to: str, url: str) -> Dict

# Send template
send_template(phone: str, template_name: str, template_language: str, variables: Dict = None, header_variables: List = None, button_variables: List = None, account_id: str = None) -> Dict

# Send interactive buttons
send_interactive_buttons(account_id: str, to: str, body_text: str, buttons: List[Dict], header_text: str = None, footer_text: str = None) -> Dict

# Send interactive list
send_interactive_list(account_id: str, to: str, body_text: str, sections: List[Dict], button_text: str, header_text: str = None, footer_text: str = None) -> Dict

# Template management
sync_templates(account_id: str) -> Dict
list_templates(account_id: str) -> List
create_template(account_id: str, template_data: Dict) -> Dict

# Contact management
get_contacts(account_id: str, limit: int = 50, offset: int = 0) -> List
get_contact(account_id: str, phone_number: str) -> Dict

# Session management
get_session_status(account_id: str, phone_number: str) -> Dict
```

### RCSChannel

```python
# Send message
send_message(phone_number: str, template_key: str, template_value: str = None) -> Dict

# Send rich message
send_rich_message(phone_number: str, content: Dict, suggestions: List = None) -> Dict

# Template management
list_templates() -> List
create_template(template_data: Dict) -> Dict
get_template(template_id: str) -> Dict

# Bot management
list_bots() -> List
create_bot(bot_data: Dict) -> Dict
register_bot(bot_id: str) -> Dict

# Session management
list_sessions(limit: int = 50, offset: int = 0) -> List
```

### VoiceChannel

```python
# LiveKit
get_livekit_token(room_name: str, participant_name: str, participant_identity: str = None) -> str

# Call routing
configure_routing(routing_config: Dict) -> Dict
get_routing_config() -> Dict

# Phone numbers
list_phone_numbers() -> List
get_phone_number(number_id: str) -> Dict
purchase_phone_number(country_code: str, capabilities: List) -> Dict

# Recordings
list_recordings(limit: int = 50, offset: int = 0) -> List
get_recording(recording_id: str) -> Dict
delete_recording(recording_id: str) -> Dict

# UAT API
proxy_uat_request(campaign_id: str, agent_id: str, user_id: str, uat_payload: Dict) -> Dict
```

### InstagramChannel

```python
# OAuth
initiate_oauth(agent_id: str, redirect_uri: str) -> Dict
complete_oauth(agent_id: str, code: str) -> Dict

# Messaging
send_message(agent_id: str, recipient_id: str, text: str = None, attachment: Dict = None) -> Dict
```

### MessengerChannel

```python
# OAuth
initiate_oauth(agent_id: str, redirect_uri: str) -> Dict
complete_oauth(agent_id: str, code: str) -> Dict

# Messaging
send_message(agent_id: str, recipient_id: str, message: str, quick_replies: List = None) -> Dict
```

### LinkedInChannel

```python
# OAuth
initiate_oauth(agent_id: str, redirect_uri: str) -> Dict
complete_oauth(agent_id: str, code: str) -> Dict

# Messaging
send_message(agent_id: str, recipient_id: str, message: str, conversation_id: str = None, account_id: str = None) -> Dict
```

---

## Services

### AgentsService

```python
# CRUD
list() -> List[Agent]
get(agent_id: str) -> Agent
create(name: str, channel: Channel, ...) -> Agent
update(agent_id: str, ...) -> Agent
delete(agent_id: str) -> Dict

# Chat
chat(bot_id: str, message: str, session_id: str = None, ...) -> AgentChatResponse
chat_voice(agent_id: str, transcript: str, user_id: str, ...) -> Dict
chat_web_widget(bot_id: str, message: str, api_key: str, ...) -> Dict
invalidate_cache(agent_id: str) -> Dict

# API Keys
generate_api_key(agent_id: str) -> AgentAPIKeyResponse
get_api_key(agent_id: str) -> AgentAPIKeyResponse
revoke_api_key(agent_id: str) -> AgentAPIKeyResponse

# Config
get_config(agent_id: str = None, phone_number: str = None, ...) -> Dict
```

### CampaignsService

```python
# WhatsApp campaigns
create_whatsapp(campaign_name: str, template_id: str, account_id: str, csv_file: BinaryIO, campaign_type: str = "marketing") -> Dict
list_whatsapp(limit: int = 50, offset: int = 0) -> List
get_whatsapp(campaign_id: str) -> Dict

# Control
pause(campaign_id: str) -> Dict
resume(campaign_id: str) -> Dict
cancel(campaign_id: str) -> Dict
get_status(campaign_id: str) -> Dict
get_analytics(campaign_id: str) -> Dict
```

### CreditsService

```python
# Balance
get_balance(organization_id: str = None, agent_id: str = None) -> CreditBalance
check_balance_with_rate(service_key: ServiceKey, quantity: int, ...) -> Dict

# Operations
reserve(organization_id: str, service_key: ServiceKey, amount: float, ...) -> Dict
settle_reservation(reservation_id: str, actual_used: float, ...) -> Dict
deduct(service_key: ServiceKey, quantity: int, ...) -> Dict
topup(organization_id: str, amount: float, ...) -> Dict
transfer(from_org_id: str, to_org_id: str, amount: float, ...) -> Dict

# Analytics
get_rate(service_key: ServiceKey, ...) -> ServiceRate
get_transactions(limit: int = 50, offset: int = 0, ...) -> List[CreditTransaction]
get_usage_analytics(month: str, ...) -> UsageAnalytics
get_service_definitions(active_only: bool = True) -> List

# Billing
list_invoices(limit: int = 50, offset: int = 0, ...) -> List[Invoice]
get_invoice(invoice_id: str) -> Invoice
download_invoice(invoice_id: str, format: str = "pdf") -> Dict
initiate_payment(...) -> Dict
```

### KnowledgeBaseService

```python
# Document processing
process_document(agent_id: str, file_path: str = None, file_content: bytes = None, filename: str = None) -> Dict
add_text(agent_id: str, text: str, metadata: Dict = None) -> Dict
add_webpage(agent_id: str, url: str, metadata: Dict = None) -> Dict

# Search
search(agent_id: str, query: str, limit: int = 5, threshold: float = 0.7) -> List[SearchResult]

# Management
list_documents(agent_id: str, limit: int = 50, offset: int = 0) -> List[Document]
delete_document(agent_id: str, document_id: str) -> Dict
clear_knowledge_base(agent_id: str) -> Dict

# Crawling
crawl_website(agent_id: str, url: str, max_pages: int = 50, ...) -> Dict
get_crawl_status(job_id: str) -> Dict
```

### UsersService

```python
list() -> List[User]
get(user_id: str) -> User
create(email: str, full_name: str, role: str = "user", ...) -> User
update(user_id: str, ...) -> User
delete(user_id: str) -> Dict
generate_api_key() -> Dict
revoke_api_key() -> Dict
```

### OrganizationsService

```python
list() -> List[Organization]
get(organization_id: str) -> Organization
create(name: str, display_name: str, org_type: str, ...) -> Organization
update(organization_id: str, ...) -> Organization
delete(organization_id: str) -> Dict
add_user(organization_id: str, user_id: str, role_key: str) -> Dict
remove_user(organization_id: str, user_id: str) -> Dict
allocate_credits(organization_id: str, amount: float, reason: str = None) -> Dict
```

### MCPToolsService

```python
list_tools(include_builtin: bool = True) -> List[Dict]
list_connections() -> List[Dict]
get_connection(connection_id: str) -> Dict
initiate_oauth(platform_name: str, connection_label: str = None, ...) -> Dict
complete_oauth(platform_name: str, code: str, state: str) -> Dict
delete_connection(connection_id: str) -> Dict
```

---

## Models

### Agent

```python
class Agent(IdentifiableModel):
    name: str
    description: Optional[str]
    channel: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    system_prompt: Optional[str]
    agent_mode: Optional[str]
    agent_type: Optional[str]
    languages: List[str]
    is_active: Optional[bool]
    is_deployed: Optional[bool]
```

### AgentChatResponse

```python
class AgentChatResponse(FonadaBaseModel):
    reply: str
    session_id: str
    llm_provider: Optional[str]
    llm_model: Optional[str]
    request_id: Optional[str]
    media: List[Any]
```

### Campaign

```python
class Campaign(IdentifiableModel):
    name: str
    status: CampaignStatus
    campaign_type: CampaignType
    template_id: Optional[str]
    account_id: Optional[str]
    total_recipients: int
    sent_count: int
    delivered_count: int
    failed_count: int
```

### CreditBalance

```python
class CreditBalance(FonadaBaseModel):
    balance: float
    reserved: float
    available: float
    currency: str
```

### Document

```python
class Document(IdentifiableModel):
    agent_id: str
    filename: Optional[str]
    file_type: FileType
    status: DocumentStatus
    chunk_count: int
    metadata: Optional[Dict[str, Any]]
```

### SearchResult

```python
class SearchResult(FonadaBaseModel):
    content: str
    score: float
    document_id: str
    metadata: Optional[Dict[str, Any]]
```

---

## Enums

### Channel

```python
class Channel(str, Enum):
    WHATSAPP = "whatsapp"
    RCS = "rcs"
    TELEPHONY = "telephony"
    INSTAGRAM = "instagram"
    MESSENGER = "messenger"
    LINKEDIN = "linkedin"
    WEB_WIDGET = "web_widget"
    CHAT_BOT = "chat_bot"
```

### LLMProvider

```python
class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    AZURE = "azure"
    AZURE_OPENAI = "azure_openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
```

### AgentMode

```python
class AgentMode(str, Enum):
    CHAT = "chat"
    FLOW = "flow"
    LLM = "llm"
```

### ServiceKey

```python
class ServiceKey(str, Enum):
    WHATSAPP_TEXT = "whatsapp_text"
    WHATSAPP_TEMPLATE = "whatsapp_template"
    WHATSAPP_MEDIA = "whatsapp_media"
    RCS_MESSAGE = "rcs_message"
    RCS_TEMPLATE = "rcs_template"
    VOICE_CALL = "voice_call"
    VOICE_TTS = "voice_tts"
    VOICE_ASR = "voice_asr"
    LLM_TOKENS = "llm_tokens"
    SMS_MESSAGE = "sms_message"
```
