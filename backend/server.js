const axios = require('axios');
const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const port = 3000;
const cors = require('cors');

app.use(express.json({ limit: '50mb' })); // For JSON payloads
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(cors({
  origin: 'http://localhost:5173'
}));

app.post('/send-to-flask', async (req, res) => {
    try {
        const { image, prompt } = req.body;
        const response = await axios.post('http://localhost:5000/chat', {
            image: image, // Ensure this is base64 encoded or similar
            prompt: prompt
        });
        res.json(response.data);
    } catch (error) {
        res.status(500).send('Error communicating with Flask server');
    }
});

app.listen(port, () => {
    console.log(`Express server listening on port ${port}`);
});
