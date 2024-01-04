const express = require('express');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const multer = require('multer');
const fs = require('fs');
const path = require('path');

dotenv.config();
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static('public'));

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/public/index.html');
});

app.get('/summarize-audio', (req, res) => {
    res.sendFile(__dirname + '/public/summarize.html');
});

// Endpoint for transcribing audio
app.post('/transcribe', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).send('No file uploaded.');
        }

        const audioData = req.file.buffer;

        // Create a temporary file to store the audio data
        const tempFilePath = path.join(__dirname, 'temp_audio.wav');
        fs.writeFileSync(tempFilePath, audioData);

        // Create a read stream from the temporary file
        const audioStream = fs.createReadStream(tempFilePath);

        const response = await openai.audio.transcriptions.create({
            model: 'whisper-1',
            file: audioStream,
        });
        fs.unlinkSync(tempFilePath);

        res.json(response);
    } catch (err) {
        console.error(err);
        res.status(500).send('Internal Server Error');
    }
});

// Endpoint for summarizing text

app.post('/summarize', upload.single('audio'), async (req, res) => {
    try {
        // Check if a file is uploaded
        if (!req.file) {
            return res.status(400).send('No file uploaded.');
        }

        const audioData = req.file.buffer;

        // Create a temporary file to store the audio data
        const tempFilePath = path.join(__dirname, 'temp_audio.wav');
        fs.writeFileSync(tempFilePath, audioData);

        // Create a read stream from the temporary file
        const audioStream = fs.createReadStream(tempFilePath);

        const response = await openai.audio.transcriptions.create({
            model: 'whisper-1',
            file: audioStream,
        });

        // Add the "Summarize the above text." to the response
        const userMessage = response.text + " Summarize the above text.";

        const completion = await openai.chat.completions.create({
            messages: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": userMessage},
            ],
            model: "gpt-3.5-turbo",
        });

        const assistantResponse = completion.choices[0].message.content;
        console.log(assistantResponse);

        // Remove the temporary file
        fs.unlinkSync(tempFilePath);

        res.json(assistantResponse);
    } catch (err) {
        console.error(err);
        res.status(500).send('Internal Server Error');
    }
});


app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
