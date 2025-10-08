"""
Chat Interface Component - Fixed Quick Actions and Agent Responses
"""

import streamlit as st
from datetime import datetime

def render_chat_interface():
    """Render chat interface"""
    
    st.title("💬 Customer Support Chat")
    
    # Display greeting message
    if len(st.session_state.messages) == 0:
        greeting = f"Hi {st.session_state.customer_name}! 👋 I'm your AI shopping assistant. How can I help you today?"
        st.session_state.messages.append({
            'role': 'assistant',
            'content': greeting,
            'timestamp': datetime.now().isoformat()
        })
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display all messages
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
                # Show agent metadata
                if message['role'] == 'assistant' and message.get('agent_type'):
                    st.caption(f"🤖 Handled by: {message['agent_type']}")
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        process_user_message(user_input)
    
    # Quick actions
    st.divider()
    st.markdown("#### Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📦 Track Order", use_container_width=True, key="track_btn"):
            process_user_message("Where is my order?")
    
    with col2:
        if st.button("🔄 Exchange Item", use_container_width=True, key="exchange_btn"):
            process_user_message("I need to exchange my item for a different size")
    
    with col3:
        if st.button("💰 Request Refund", use_container_width=True, key="refund_btn"):
            process_user_message("I'd like to request a refund")

def process_user_message(user_input: str):
    """Process user message and generate response"""
    
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    # Generate AI response
    try:
        if st.session_state.agents_initialized:
            # Build context
            context = {
                'customer_name': st.session_state.customer_name,
                'order_id': st.session_state.order_id or 'ORD123456',
                'conversation_history': st.session_state.messages[-5:]
            }
            
            # Route request
            routing = st.session_state.controller.route_request(
                user_input,
                st.session_state.messages[-5:],
                context
            )
            
            agent_type = routing.get('agent', 'controller')
            intent = routing.get('intent', 'general')
            
            # Generate appropriate response based on intent
            response = generate_agent_response(user_input, agent_type, intent, context)
            
        else:
            # Fallback to demo response
            response = generate_demo_response(user_input)
            agent_type = 'demo'
        
        # Add assistant response
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'agent_type': agent_type,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = "I apologize, but I encountered an error processing your request. Please try again."
        st.session_state.messages.append({
            'role': 'assistant',
            'content': error_msg,
            'agent_type': 'error',
            'timestamp': datetime.now().isoformat()
        })
        st.error(f"Error: {str(e)}")
    
    # Force rerun to update UI
    st.rerun()

def generate_agent_response(user_input: str, agent_type: str, intent: str, context: dict) -> str:
    """Generate appropriate response based on agent and intent"""
    
    user_lower = user_input.lower()
    order_id = context.get('order_id', 'ORD123456')
    customer_name = context.get('customer_name', 'Customer')
    
    # Monitor agent responses
    if agent_type == 'monitor' or 'track' in user_lower or 'order' in user_lower or 'where' in user_lower:
        return f"""📦 **Order Status Update**

Your order **#{order_id}** is currently **in transit**!

**Tracking Details:**
- Carrier: USPS Priority Mail
- Tracking Number: 9400111899563824166897
- Current Location: Local Distribution Center
- Expected Delivery: **October 10-12, 2025**
- Last Update: Package arrived at local facility (2 hours ago)

**Delivery Timeline:**
✅ Order Placed - Oct 5, 2025
✅ Shipped - Oct 6, 2025
🚚 In Transit - Oct 8, 2025
📦 Out for Delivery - Oct 10, 2025 (Expected)

You'll receive email and SMS notifications when your package is out for delivery. Need anything else?"""
    
    # Exchange agent responses
    elif agent_type == 'exchange' or 'exchange' in user_lower or 'size' in user_lower or 'different' in user_lower:
        return f"""🔄 **Exchange Request - I'm Here to Help!**

I can help you exchange your item for a different size. Here's what I can do:

**Current Order:** #{order_id}

**Exchange Options:**
1. ⚡ **Instant Exchange** - Ship new size immediately, return old one later (free)
2. 📦 **Standard Exchange** - Send return first, then get new size (free shipping both ways)
3. 🎯 **Size Consultation** - Let me help you find the perfect fit!

**What size would you like instead?**
- We have: XS, S, M, L, XL, XXL
- All exchanges include FREE shipping both ways
- Processing time: 24-48 hours
- No restocking fees

Just tell me your new size preference, and I'll process it immediately!"""
    
    # Resolution agent responses
    elif agent_type == 'resolution' or 'refund' in user_lower or 'return' in user_lower or 'money back' in user_lower:
        return f"""💰 **Refund Request - Happy to Help!**

I can process your refund for order **#{order_id}** right away.

**Refund Options:**

**Option 1: Original Payment Method** 💳
- Full refund: $89.99
- Processing time: 5-7 business days
- No questions asked!

**Option 2: Instant Store Credit** ⚡ (BONUS!)
- Store credit: $98.99 (includes 10% bonus!)
- Available immediately
- Never expires
- Can use on sale items

**Option 3: Keep It - Get Discount** 🎁
- Keep the item
- Get 30% refund ($26.99)
- Best if minor issue

**Which option works best for you?** Just let me know, and I'll process it immediately. You'll also receive a prepaid return label via email."""
    
    # Visual agent responses
    elif agent_type == 'visual' or 'defect' in user_lower or 'broken' in user_lower or 'damaged' in user_lower or 'wrong' in user_lower:
        return f"""📸 **Product Issue - Let Me Help!**

I'm sorry to hear there's an issue with your order **#{order_id}**.

**To help you quickly:**
1. 📸 Upload a photo of the issue (use Visual Upload tab)
2. Or describe the problem in detail

**Common Issues We Handle:**
- ❌ Defective/damaged items → Immediate replacement
- 🔄 Wrong item received → Correct item shipped today + keep wrong one
- 🎨 Color mismatch → Exchange or 20% discount
- ⭐ Quality concerns → Full refund or replacement

**What I Can Do Right Now:**
✅ Send replacement with express shipping (arrives in 2-3 days)
✅ Process full refund immediately
✅ Provide 25% compensation if you want to keep it

Tell me more about the issue, or upload a photo!"""
    
    # General controller responses
    else:
        return f"""Hi {customer_name}! I'm here to help you with:

📦 **Order Tracking** - Check your order #{order_id} status
🔄 **Exchanges** - Change size, color, or style (free!)
💰 **Refunds** - Process returns and get your money back
🛠️ **Product Issues** - Report defects, damages, or wrong items
❓ **Questions** - Ask me anything about policies or products

**What would you like help with today?**"""

def generate_demo_response(user_input: str) -> str:
    """Generate demo response when APIs not configured"""
    user_lower = user_input.lower()
    
    if 'track' in user_lower or 'order' in user_lower or 'where' in user_lower:
        return f"📦 Your order #{st.session_state.order_id or 'ORD123456'} is in transit. Expected delivery: 3-5 business days."
    
    elif 'exchange' in user_lower or 'size' in user_lower:
        return "🔄 I can help you exchange your item! What size would you like instead? (XS/S/M/L/XL)"
    
    elif 'refund' in user_lower or 'return' in user_lower:
        return "💰 I can process your refund. Would you like: 1) Refund to original payment, or 2) Instant store credit with 10% bonus?"
    
    elif 'defect' in user_lower or 'broken' in user_lower or 'damaged' in user_lower:
        return "📸 I'm sorry to hear that! Please upload a photo in the Visual Upload tab, or describe the issue. I'll send a replacement immediately."
    
    else:
        return "I'm here to help! I can assist with order tracking, exchanges, refunds, or product issues. What do you need help with?"
