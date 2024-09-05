import './App.css';
import { useState } from 'react';
import axios from 'axios';

function App() {
    const [image, setImage] = useState('');
    const [prompt, setPrompt] = useState('');
    const [messages, setMessages] = useState([]);

    // Function to handle file input change
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64Image = reader.result;
                setImage(base64Image); // Set base64 string as the image state
                // Add the image to the messages array
                setMessages((prevMessages) => [...prevMessages, { type: 'image', content: base64Image, sender: 'user' }]);
            };
            reader.readAsDataURL(file); // Convert the file to base64
        }
    };

    const handleSubmit = async () => {
        // Add the user's prompt to the messages array
        setMessages((prevMessages) => [...prevMessages, { type: 'text', content: prompt, sender: 'user' }]);
        try {
            const result = await axios.post('http://localhost:3000/send-to-flask', {
                image: image, // Base64 encoded image
                prompt: prompt
            });
            // Add the response to the messages array
            setMessages((prevMessages) => [...prevMessages, { type: 'text', content: result.data.response, sender: 'bot' }]);
        } catch (error) {
            console.error('Error:', error);
        }
        setPrompt(''); // Clear the input after sending
    };

    return (
        <div className="chat-container">
            <div className="chat-window">
                {messages.map((message, index) => (
                    <div
                        key={index}
                        className={`chat-message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
                    >
                        {message.type === 'text' ? (
                            message.content
                        ) : (
                            <img src={message.content} alt="Sent" className="chat-image" />
                        )}
                    </div>
                ))}
            </div>
            <div className="chat-input-container">
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="file-input"
                />
                <input
                    type="text"
                    placeholder="Type a message..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="text-input"
                />
                <button onClick={handleSubmit} className="send-button">Send</button>
            </div>
        </div>
    );
}

export default App;
