// Generate session ID
let sessionId = localStorage.getItem('chatbot_session_id');
if (!sessionId) {
    sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('chatbot_session_id', sessionId);
}

// Load conversation history
async function loadHistory() {
    try {
        const response = await fetch(`/api/history/${sessionId}/`);
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
        
        const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        
        hideTypingIndicator();
        
        if (data.error) {
            addMessage(`Error: ${data.error}`, 'ai');
        } else {
            addMessage(data.ai_response, 'ai');
            if (data.session_id) {
                sessionId = data.session_id;
                localStorage.setItem('chatbot_session_id', sessionId);
            }
        }
    } catch (error) {
        hideTypingIndicator();
        addMessage('Failed to send message. Please try again.', 'ai');
        console.error('Error:', error);
    }
}

// Get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Event listeners
document.getElementById('sendButton').addEventListener('click', sendMessage);

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

