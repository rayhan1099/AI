// Chat application state
let isTyping = false;
let messageCount = 0;

// Format timestamp
function formatTime(date = new Date()) {
    return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
    });
}

// Update character count
function updateCharCount() {
    const input = document.getElementById('messageInput');
    const charCount = document.getElementById('charCount');
    const count = input.value.length;
    charCount.textContent = count;
    
    if (count > 900) {
        charCount.parentElement.style.color = '#f5576c';
    } else {
        charCount.parentElement.style.color = '';
    }
}

// Load conversation history
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        const chatMessages = document.getElementById('chatMessages');
        const welcomeMessage = chatMessages.querySelector('.welcome-message');
        
        if (data.history && data.history.length > 0) {
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
            
            data.history.forEach(msg => {
                addMessage(msg.content, msg.role);
            });
            
            scrollToBottom();
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Add message to chat
function addMessage(message, type) {
    const chatMessages = document.getElementById('chatMessages');
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    
    // Remove welcome message if it exists
    if (welcomeMessage && messageCount === 0) {
        welcomeMessage.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    // Create avatar
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = type === 'user' ? 'U' : 'AI';
    messageDiv.appendChild(avatarDiv);
    
    // Create content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const p = document.createElement('p');
    p.textContent = message;
    contentDiv.appendChild(p);
    
    // Add timestamp
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = formatTime();
    contentDiv.appendChild(timeDiv);
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    messageCount++;
    scrollToBottom();
}

// Scroll to bottom smoothly
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

// Show typing indicator
function showTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    indicator.style.display = 'block';
    scrollToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    indicator.style.display = 'none';
}

// Send message
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const message = input.value.trim();
    
    if (!message || isTyping) {
        return;
    }
    
    if (message.length > 1000) {
        alert('Message is too long. Maximum 1000 characters.');
        return;
    }
    
    try {
        isTyping = true;
        sendButton.disabled = true;
        
        showTypingIndicator();
        addMessage(message, 'user');
        input.value = '';
        updateCharCount();
        
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
        addMessage('Failed to send message. Please check your connection and try again.', 'ai');
        console.error('Error:', error);
    } finally {
        isTyping = false;
        sendButton.disabled = false;
        input.focus();
    }
}

// Clear history
async function clearHistory() {
    if (!confirm('Are you sure you want to clear the conversation history?')) {
        return;
    }
    
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
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                        </svg>
                    </div>
                    <h2>Welcome! 👋</h2>
                    <p>I'm your AI assistant. How can I help you today?</p>
                </div>
            `;
            messageCount = 0;
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        alert('Failed to clear history. Please try again.');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');
    const messageInput = document.getElementById('messageInput');
    
    // Send button click
    sendButton.addEventListener('click', sendMessage);
    
    // Clear button click
    clearButton.addEventListener('click', clearHistory);
    
    // Enter key to send
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Character count update
    messageInput.addEventListener('input', updateCharCount);
    
    // Auto-resize input (optional enhancement)
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
    
    // Load history on page load
    loadHistory();
    messageInput.focus();
    
    // Prevent form submission on Enter in input
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
        }
    });
});

// Handle visibility change (resume focus when tab becomes visible)
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        const input = document.getElementById('messageInput');
        input.focus();
    }
});
