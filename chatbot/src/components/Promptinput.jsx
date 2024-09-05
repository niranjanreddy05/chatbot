import React from 'react';
import './PromptInput.css'; // Custom CSS for styling

const PromptInput = ({ prompt, onPromptChange }) => {
    return (
        <div className="prompt-input-container">
            <input
                type="text"
                placeholder="Enter your prompt here..."
                value={prompt}
                onChange={onPromptChange}
                className="prompt-input"
            />
        </div>
    );
};

export default PromptInput;
