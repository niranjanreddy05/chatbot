import React from 'react';
import './Fileinput.css'; // Custom CSS for styling

const FileInput = ({ onFileChange }) => {
    return (
        <div className="file-input-container">
            <input
                type="file"
                accept="image/*"
                onChange={onFileChange}
                className="file-input"
            />
        </div>
    );
};

export default FileInput;
