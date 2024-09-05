import React from 'react';
import './ResponseDisplay.css'; // Custom CSS for styling

const ResponseDisplay = ({ response }) => {
    return (
        <div className="response-display-container">
            <p className="response-text">Response: {response}</p>
        </div>
    );
};

export default ResponseDisplay;
