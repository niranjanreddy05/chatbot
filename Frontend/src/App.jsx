import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [showChat, setShowChat] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const chatMessagesRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    async function warmUp() {
      try {
        const response = await axios.get("http://localhost:3000/warmup");
        console.log(response);
      } catch (error) {
        console.error("Error warming up:", error);
      }
    }

    warmUp();
  }, []);

  const addMessage = (content, isUser = false, isLoading = false) => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { content, isUser, isLoading },
    ]);
  };

  const sendMessage = async (message = "", file = null) => {
    if (message || file) {
      addMessage(file || message, true);

      let image = null;
      if (file) {
        image = await convertToBase64(file);
      }

      addMessage(null, false, true);

      try {
        const response = await axios.post(
          "http://localhost:3000/send-to-flask",
          {
            prompt: message,
            image: image,
          }
        );

        setMessages((prevMessages) =>
          prevMessages.filter((msg) => !msg.isLoading)
        );

        if (response.data && response.data.response) {
          addMessage(response.data.response, false);
        } else {
          addMessage("Error: Unexpected response format from server", false);
        }
      } catch (error) {
        setMessages((prevMessages) =>
          prevMessages.filter((msg) => !msg.isLoading)
        );
        addMessage("Error: " + (error.response?.data || error.message), false);
      }
    }
  };

  const convertToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });
  };

  const handleSendClick = () => {
    if (inputValue.trim()) {
      sendMessage(inputValue);
      setInputValue("");
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleFile = (file) => {
    sendMessage("", file);
    setShowChat(true);
    setShowModal(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSendClick();
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  };

  const toggleModal = () => {
    setShowModal(!showModal);
  };

  const closeModal = () => {
    setShowModal(false);
  };

  return (
    <>
      <div className={`app-container ${showModal ? "blur-background" : ""}`}>
        <div className="navbar">
          <div className="chatbot-title">AI Chatbot</div>
        </div>
        <div className="content">
          {!showChat ? (
            <div className="initial-view">
              <div className="main-heading">Hello, how can I help you?</div>
              <div
                className={`drop-zone ${isDragging ? "dragging" : ""}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <p className="prompt">Drag and drop an image here</p>
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={handleFileChange}
                />
                <button onClick={() => fileInputRef.current.click()}>
                  Or click to upload
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="chat-container show">
                <div className="chat-messages" ref={chatMessagesRef}>
                  {messages.map((msg, index) => (
                    <div
                      key={index}
                      className={`message ${
                        msg.isUser ? "user-message" : "bot-message"
                      }`}
                    >
                      {msg.isLoading ? (
                        <div className="loading-dots">
                          <div className="dot"></div>
                          <div className="dot"></div>
                          <div className="dot"></div>
                        </div>
                      ) : msg.content instanceof File ? (
                        <img
                          src={URL.createObjectURL(msg.content)}
                          alt="User upload"
                          className="uploaded-image"
                        />
                      ) : typeof msg.content === "string" ? (
                        msg.content
                      ) : (
                        JSON.stringify(msg.content)
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
                  <button className="btn-upload" onClick={toggleModal}>
                    <i className="fas fa-upload"></i>
                  </button>
                  <button className="btn-send" onClick={handleSendClick}>
                    ➤
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
      {showModal && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div
              className={`drop-zone ${isDragging ? "dragging" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <button className="modal-close" onClick={closeModal}>
                ×
              </button>
              <p>Drag and drop an image here</p>
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              <button onClick={() => fileInputRef.current.click()}>
                Or click to upload
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
