// Load conversation history
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = '';
            
            data.history.forEach(msg => {
                addMessage(msg.content, msg.role);
            });
        }
    } catch (error) {
        console.error('Error loading history:', error);
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
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) {
        return;
    }
    
    try {
        showTypingIndicator();
        addMessage(message, 'user');
        input.value = '';
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        const data = await response.json();
        
        hideTypingIndicator();
        
        if (data.error) {
            addMessage(`Error: ${data.error}`, 'ai');
        } else {
            addMessage(data.ai_response, 'ai');
        }
    } catch (error) {
        hideTypingIndicator();
        addMessage('Failed to send message. Please try again.', 'ai');
        console.error('Error:', error);
    }
}

// Clear history
async function clearHistory() {
    try {
        const response = await fetch('/api/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.message) {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = '';
            addMessage('Hello! I\'m your AI assistant. How can I help you today?', 'ai');
        }
    } catch (error) {
        console.error('Error clearing history:', error);
    }
}

// Event listeners
document.getElementById('sendButton').addEventListener('click', sendMessage);
document.getElementById('clearButton').addEventListener('click', clearHistory);

document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Load history on page load
window.addEventListener('load', () => {
    loadHistory();
    document.getElementById('messageInput').focus();
});

