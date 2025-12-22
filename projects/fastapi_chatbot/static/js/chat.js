// Generate unique client ID
const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
let ws = null;

// Initialize WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('Connection error. Please refresh the page.');
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        // Try to reconnect after 3 seconds
        setTimeout(initWebSocket, 3000);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'user_message':
            addMessage(data.message, 'user');
            break;
        case 'ai_message':
            hideTypingIndicator();
            addMessage(data.message, 'ai');
            break;
        case 'typing':
            if (data.status) {
                showTypingIndicator();
            } else {
                hideTypingIndicator();
            }
            break;
    }
}

// Add message to chat
function addMessage(message, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const p = document.createElement('p');
    p.textContent = message;
    
    contentDiv.appendChild(p);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    document.getElementById('typingIndicator').style.display = 'flex';
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    document.getElementById('typingIndicator').style.display = 'none';
}

// Send message
function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) {
        return;
    }
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ message: message }));
        input.value = '';
    } else {
        // Fallback to REST API if WebSocket is not available
        sendMessageViaAPI(message);
    }
}

// Fallback: Send message via REST API
async function sendMessageViaAPI(message) {
    try {
        showTypingIndicator();
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: clientId
            })
        });
        
        const data = await response.json();
        
        hideTypingIndicator();
        addMessage(message, 'user');
        addMessage(data.ai_response, 'ai');
    } catch (error) {
        hideTypingIndicator();
        showError('Failed to send message. Please try again.');
        console.error('Error:', error);
    }
}

// Show error message
function showError(message) {
    addMessage(`Error: ${message}`, 'ai');
}

// Event listeners
document.getElementById('sendButton').addEventListener('click', sendMessage);

document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Initialize WebSocket on page load
window.addEventListener('load', () => {
    initWebSocket();
});

// Focus input on load
document.getElementById('messageInput').focus();

