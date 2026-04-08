# Error Handling

The Fonada SDK provides detailed exceptions for different error scenarios.

## Exception Hierarchy

```
FonadaError (base)
├── AuthenticationError      # 401 - Invalid credentials
├── AuthorizationError       # 403 - Access denied
├── NotFoundError           # 404 - Resource not found
├── ValidationError         # 400/422 - Invalid input
├── RateLimitError          # 429 - Too many requests
├── APIError                # 5xx - Server errors
├── InsufficientCreditsError # 402 - Not enough credits
├── MessagingWindowExpiredError  # WhatsApp 24h window
├── TemplateNotApprovedError     # Template not approved
├── InvalidPhoneNumberError      # Invalid phone format
├── ConnectionError         # Network issues
├── TimeoutError           # Request timeout
└── ConfigurationError     # SDK configuration issues
```

## Basic Error Handling

```python
from fonada import FonadaClient
from fonada.exceptions import (
    FonadaError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    InsufficientCreditsError,
    MessagingWindowExpiredError,
)

client = FonadaClient(...)

try:
    response = client.whatsapp.send_text(
        account_id="uuid",
        to="+919876543210",
        text="Hello!"
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
    # Handle: Check API key, refresh JWT token
    
except AuthorizationError as e:
    print(f"Access denied: {e.message}")
    # Handle: Check permissions
    
except NotFoundError as e:
    print(f"Resource not found: {e.message}")
    # Handle: Check IDs, resource exists
    
except ValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"Details: {e.errors}")
    # Handle: Fix input parameters
    
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
    print(f"Retry after: {e.retry_after} seconds")
    # Handle: Wait and retry
    
except InsufficientCreditsError as e:
    print(f"Not enough credits: {e.message}")
    # Handle: Top up credits
    
except MessagingWindowExpiredError as e:
    print(f"Messaging window expired: {e.message}")
    # Handle: Use template message instead
    
except FonadaError as e:
    print(f"API error: {e.message}")
    print(f"Status code: {e.status_code}")
    print(f"Response: {e.response}")
```

## Exception Details

### AuthenticationError

Raised when authentication fails (401).

```python
from fonada.exceptions import AuthenticationError

try:
    client = FonadaClient(api_key="invalid_key", project_url="...")
    client.agents.list()
except AuthenticationError as e:
    print(f"Message: {e.message}")
    print(f"Status: {e.status_code}")  # 401
```

**Common causes:**
- Invalid API key
- Expired JWT token
- Missing authentication header

### ValidationError

Raised when input validation fails (400/422).

```python
from fonada.exceptions import ValidationError

try:
    client.whatsapp.send_text(
        account_id="not-a-uuid",
        to="invalid-phone",
        text=""
    )
except ValidationError as e:
    print(f"Message: {e.message}")
    print(f"Errors: {e.errors}")  # Dict with field-specific errors
```

**Common causes:**
- Invalid UUID format
- Invalid phone number
- Missing required fields
- Invalid enum values

### RateLimitError

Raised when rate limit is exceeded (429).

```python
from fonada.exceptions import RateLimitError
import time

try:
    for i in range(1000):
        client.whatsapp.send_text(...)
except RateLimitError as e:
    print(f"Rate limited! Retry after {e.retry_after}s")
    time.sleep(e.retry_after)
    # Retry the request
```

### InsufficientCreditsError

Raised when credit balance is too low (402).

```python
from fonada.exceptions import InsufficientCreditsError

try:
    campaign = client.campaigns.create_whatsapp(...)
except InsufficientCreditsError as e:
    print(f"Not enough credits: {e.message}")
    # Prompt user to top up
    balance = client.credits.get_balance()
    print(f"Current balance: {balance.balance}")
```

### MessagingWindowExpiredError

Raised when WhatsApp 24-hour window has closed.

```python
from fonada.exceptions import MessagingWindowExpiredError

try:
    client.whatsapp.send_text(
        account_id="uuid",
        to="+919876543210",
        text="Hello!"
    )
except MessagingWindowExpiredError as e:
    print("24-hour window expired, using template...")
    client.whatsapp.send_template(
        phone="+919876543210",
        template_name="re_engagement",
        template_language="en"
    )
```

## Retry Logic

The SDK includes automatic retry for transient errors:

```python
# SDK automatically retries on:
# - 429 Rate Limit (waits for Retry-After)
# - 5xx Server errors (exponential backoff)
# - Connection errors (up to max_retries)

client = FonadaClient(
    api_key="fsk_xxx",
    project_url="https://xxx.supabase.co",
    max_retries=5  # Default is 3
)
```

## Custom Error Handling Pattern

```python
from fonada import FonadaClient
from fonada.exceptions import FonadaError
import logging

logger = logging.getLogger(__name__)

def send_message_safe(client, account_id, to, text):
    """Send message with comprehensive error handling."""
    try:
        return client.whatsapp.send_text(
            account_id=account_id,
            to=to,
            text=text
        )
    except AuthenticationError:
        logger.error("Authentication failed - check API credentials")
        raise
    except InsufficientCreditsError:
        logger.error("Insufficient credits - topping up required")
        # Could trigger auto top-up here
        raise
    except MessagingWindowExpiredError:
        logger.warning("Window expired - falling back to template")
        return client.whatsapp.send_template(
            phone=to,
            template_name="default_message",
            template_language="en"
        )
    except RateLimitError as e:
        logger.warning(f"Rate limited - retry after {e.retry_after}s")
        raise
    except FonadaError as e:
        logger.error(f"API error: {e.message} (status: {e.status_code})")
        raise
```

## Error Response Format

All exceptions include:

```python
exception.message      # Human-readable error message
exception.status_code  # HTTP status code
exception.response     # Full API response dict
exception.error_code   # Specific error code (if available)
```

Example API error response:
```json
{
    "success": false,
    "error": "Messaging window expired",
    "error_code": "MESSAGING_WINDOW_EXPIRED",
    "details": {
        "window_expired_at": "2026-03-23T10:00:00Z"
    }
}
```
