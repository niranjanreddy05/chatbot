import React from 'react';
import './SubmitButton.css'; // Custom CSS for styling

const SubmitButton = ({ onClick }) => {
    return (
        <div className="submit-button-container">
            <button onClick={onClick} className="submit-button">
                Send
            </button>
        </div>
    );
};

export default SubmitButton;
