# WhatsApp Channel

Complete guide for WhatsApp Business API integration.

## Overview

The WhatsApp channel provides full access to WhatsApp Business API features including:
- Text, media, and interactive messages
- Template messages for notifications
- Contact and session management
- Template creation and synchronization

## Prerequisites

- WhatsApp Business Account
- Phone number registered with WhatsApp Business
- Account ID (UUID) from Fonada platform

## Sending Messages

### Text Messages

```python
response = client.whatsapp.send_text(
    account_id="your-account-uuid",
    to="+919876543210",
    text="Hello! How can I help you today?"
)
print(f"Message ID: {response.get('message_id')}")
```

### Image Messages

```python
response = client.whatsapp.send_image(
    account_id="your-account-uuid",
    to="+919876543210",
    url="https://example.com/image.jpg",
    caption="Check out this product!"
)
```

### Video Messages

```python
response = client.whatsapp.send_video(
    account_id="your-account-uuid",
    to="+919876543210",
    url="https://example.com/video.mp4",
    caption="Watch our demo"
)
```

### Document Messages

```python
response = client.whatsapp.send_document(
    account_id="your-account-uuid",
    to="+919876543210",
    url="https://example.com/invoice.pdf",
    filename="invoice_march_2026.pdf",
    caption="Your invoice for March 2026"
)
```

### Audio Messages

```python
response = client.whatsapp.send_audio(
    account_id="your-account-uuid",
    to="+919876543210",
    url="https://example.com/audio.mp3"
)
```

## Template Messages

Templates are required for initiating conversations (outside 24-hour window).

### Send Template

```python
response = client.whatsapp.send_template(
    phone="+919876543210",
    template_name="order_confirmation",
    template_language="en",
    variables={
        "customer_name": "John",
        "order_id": "ORD-12345",
        "amount": "500"
    },
    header_variables=["https://example.com/product.jpg"],
    button_variables=[
        {"type": "url", "value": "track/ORD-12345"}
    ],
    account_id="your-account-uuid"
)
```

### Variable Types

```python
# Body variables (numbered in template)
variables = {
    "1": "John",      # {{1}}
    "2": "ORD-12345", # {{2}}
    "3": "500"        # {{3}}
}

# Or named variables
variables = {
    "customer_name": "John",
    "order_id": "ORD-12345"
}

# Header variables (for media headers)
header_variables = [
    "https://example.com/image.jpg"  # For image header
]

# Button variables (for URL buttons with dynamic suffix)
button_variables = [
    {"type": "url", "value": "order/12345"}
]
```

## Interactive Messages

### Button Message

```python
response = client.whatsapp.send_interactive_buttons(
    account_id="your-account-uuid",
    to="+919876543210",
    body_text="How would you like to proceed?",
    buttons=[
        {"id": "confirm", "title": "Confirm Order"},
        {"id": "cancel", "title": "Cancel Order"},
        {"id": "help", "title": "Need Help"}
    ],
    header_text="Order Confirmation",
    footer_text="Reply within 24 hours"
)
```

### List Message

```python
response = client.whatsapp.send_interactive_list(
    account_id="your-account-uuid",
    to="+919876543210",
    body_text="Select a category to browse:",
    button_text="View Categories",
    sections=[
        {
            "title": "Electronics",
            "rows": [
                {"id": "phones", "title": "Phones", "description": "Latest smartphones"},
                {"id": "laptops", "title": "Laptops", "description": "Work & gaming laptops"}
            ]
        },
        {
            "title": "Fashion",
            "rows": [
                {"id": "men", "title": "Men's Wear", "description": "Shirts, pants, etc."},
                {"id": "women", "title": "Women's Wear", "description": "Dresses, tops, etc."}
            ]
        }
    ],
    header_text="Product Categories",
    footer_text="Tap to select"
)
```

## Template Management

### Sync Templates

Synchronize templates from WhatsApp Business API:

```python
result = client.whatsapp.sync_templates(account_id="your-account-uuid")
print(f"Synced {result.get('count')} templates")
```

### List Templates

```python
templates = client.whatsapp.list_templates(account_id="your-account-uuid")
for template in templates:
    print(f"Name: {template['name']}")
    print(f"Status: {template['status']}")
    print(f"Category: {template['category']}")
    print()
```

### Create Template

```python
template = client.whatsapp.create_template(
    account_id="your-account-uuid",
    template_data={
        "name": "order_shipped",
        "language": "en",
        "category": "UTILITY",
        "components": [
            {
                "type": "HEADER",
                "format": "TEXT",
                "text": "Order Shipped!"
            },
            {
                "type": "BODY",
                "text": "Hi {{1}}, your order {{2}} has been shipped. Track at {{3}}"
            },
            {
                "type": "FOOTER",
                "text": "Thank you for shopping with us"
            },
            {
                "type": "BUTTONS",
                "buttons": [
                    {
                        "type": "URL",
                        "text": "Track Order",
                        "url": "https://example.com/track/{{1}}"
                    }
                ]
            }
        ]
    }
)
```

## Contact Management

### Get Contacts

```python
contacts = client.whatsapp.get_contacts(
    account_id="your-account-uuid",
    limit=50,
    offset=0
)
for contact in contacts:
    print(f"Phone: {contact['phone_number']}")
    print(f"Name: {contact.get('name', 'Unknown')}")
    print(f"Last message: {contact.get('last_message_at')}")
```

### Get Single Contact

```python
contact = client.whatsapp.get_contact(
    account_id="your-account-uuid",
    phone_number="+919876543210"
)
print(f"Contact: {contact}")
```

## Session Management

WhatsApp has a 24-hour messaging window. Check session status before sending non-template messages.

### Check Session Status

```python
session = client.whatsapp.get_session_status(
    account_id="your-account-uuid",
    phone_number="+919876543210"
)

if session.get("is_active"):
    # Can send regular messages
    client.whatsapp.send_text(...)
else:
    # Must use template
    client.whatsapp.send_template(...)
```

## Error Handling

```python
from fonada.exceptions import (
    MessagingWindowExpiredError,
    TemplateNotApprovedError,
    InvalidPhoneNumberError,
    InsufficientCreditsError
)

try:
    response = client.whatsapp.send_text(
        account_id="uuid",
        to="+919876543210",
        text="Hello!"
    )
except MessagingWindowExpiredError:
    # 24-hour window expired
    client.whatsapp.send_template(
        phone="+919876543210",
        template_name="re_engage",
        template_language="en"
    )
except TemplateNotApprovedError as e:
    print(f"Template not approved: {e.message}")
except InvalidPhoneNumberError as e:
    print(f"Invalid phone number: {e.message}")
except InsufficientCreditsError:
    print("Not enough credits")
```

## Best Practices

1. **Always check session status** before sending non-template messages
2. **Use templates** for notifications, reminders, and re-engagement
3. **Validate phone numbers** before sending (use `utils.validators`)
4. **Handle errors gracefully** with fallback to templates
5. **Monitor credits** for high-volume messaging
6. **Use meaningful template names** for easy identification

## Rate Limits

WhatsApp Business API has rate limits based on your tier:
- Tier 1: 1,000 business-initiated conversations/day
- Tier 2: 10,000 business-initiated conversations/day
- Tier 3: 100,000 business-initiated conversations/day
- Tier 4: Unlimited

The SDK handles rate limiting automatically with retry logic.
