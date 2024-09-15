// App.js
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [showChat, setShowChat] = useState(false);
  const chatMessagesRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  const addMessage = (content, isUser = false) => {
    setMessages(prevMessages => [...prevMessages, { content, isUser }]);
  };

  const sendMessage = async (message = '', file = null) => {
    if (message || file) {
      if (!showChat) setShowChat(true);
      addMessage(file || message, true);

      let image = null;
      if (file) {
        image = await convertToBase64(file);
      }

      addMessage('Bot is typing...', false);

      try {
        const response = await axios.post('http://localhost:3000/send-to-flask', {
          prompt: message,
          image: image
        });

        setMessages(prevMessages => prevMessages.filter(msg => msg.content !== 'Bot is typing...'));

        if (response.data && response.data.response) {
          addMessage(response.data.response, false);
        } else {
          addMessage('Error: Unexpected response format from server', false);
        }
      } catch (error) {
        setMessages(prevMessages => prevMessages.filter(msg => msg.content !== 'Bot is typing...'));
        addMessage('Error: ' + (error.response?.data || error.message), false);
      }
    }
  };

  const convertToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };

  const handleSendClick = () => {
    if (inputValue.trim()) {
      sendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      sendMessage('', file);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendClick();
    }
  };

  return (
    <div className="app-container">
      <div className="chatbot-title">AI Chatbot</div>

      {!showChat && (
        <div className="main-heading">Hello, how can I help you?</div>
      )}

      <div className={`chat-container ${showChat ? 'show' : ''}`}>
        <div className="chat-messages" ref={chatMessagesRef}>
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.isUser ? 'user-message' : 'bot-message'}`}>
              {msg.content instanceof File ? (
                <img src={URL.createObjectURL(msg.content)} alt="User upload" />
              ) : (
                typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
              )}
            </div>
          ))}
        </div>
      </div>
      
      <div className="user-input-container">
        <div className="user-input">
          <input
            type="text"
            className="form-control"
            placeholder="Type your message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button className="btn-upload" onClick={() => fileInputRef.current.click()}>
            <i className="fas fa-upload"></i>
          </button>
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          <button className="btn-send" onClick={handleSendClick}>➤</button>
        </div>
      </div>
    </div>
  );
}

export default App;