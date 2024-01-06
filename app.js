const express = require('express');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const RapidAPI = require('rapidapi-connect');

const { PDFLoader } = require("langchain/document_loaders/fs/pdf");

const { CSVLoader } = require("langchain/document_loaders/fs/csv");

const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

const { OpenAIEmbeddings } = require("langchain/embeddings/openai");

const { ChatOpenAI } = require("langchain/chat_models/openai");

const { FaissStore } = require("langchain/vectorstores/faiss");

const {
    RunnablePassthrough,
    RunnableSequence,
} = require("langchain/schema/runnable");

const { StringOutputParser } = require("langchain/schema/output_parser");
const {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} = require("langchain/prompts");

const { formatDocumentsAsString } = require("langchain/util/document");

dotenv.config();
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const app = express();
const port = process.env.PORT || 3000;
const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/public/index.html');
});

app.get('/summarize-audio', (req, res) => {
    res.sendFile(__dirname + '/public/summarize-audio.html');
});

app.get('/summarize-pdf', (req, res) => {
    res.sendFile(__dirname + '/public/summarize-pdf.html');
});

app.get('/summarize-yt', (req, res) => {
    res.sendFile(__dirname + '/public/summarize-yt.html');
});

app.get('/summarize-csv', (req, res) => {
    res.sendFile(__dirname + '/public/summarize-csv.html');
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

app.post('/summarize-audio', upload.single('audio'), async (req, res) => {
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

app.post('/pdf', upload.single('pdf'),async (req, res) => {
    try {
        const model = new ChatOpenAI({
            modelName: "gpt-3.5-turbo"
        })
        if (!req.file) {
            return res.status(400).send('No PDF file uploaded.');
        }
        
        const pdfFilePath = path.join(__dirname, 'uploads', req.file.originalname);
        fs.writeFileSync(pdfFilePath, req.file.buffer);

        // Step-1 Load PDF

        // provide the pdf file path here
        // const loader = new PDFLoader("./load_file_2B.csv");
        const loader = new PDFLoader(pdfFilePath);
        const docs = await loader.load();
        
        // Step-2 Split the pdf into chunks

        //splitter function
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 20,
        });

        // created chunks from pdf
        const splittedDocs = await splitter.splitDocuments(docs);
        // we will use OpenAI's embedding models
        const embeddings = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY// In Node.js defaults to process.env.OPENAI_API_KEY
            // batchSize: 512, // Default value if omitted is 512. Max is 2048
        });
        // Create a vector store from the documents.
        const vectorStore = await FaissStore.fromDocuments(
            splittedDocs,
            embeddings
        );

        // Initialize a retriever wrapper around the vector store.
        const vectorStoreRetriever = vectorStore.asRetriever();

        // Create a system & human prompt for the chat model.
        const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    {context}`;
        const messages = [
            SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.fromTemplate("{question}"),
        ];
        const prompt = ChatPromptTemplate.fromMessages(messages);

        // Construct the runnable chain.
        const chain = RunnableSequence.from([
            {
                context: vectorStoreRetriever.pipe(formatDocumentsAsString),
                question: new RunnablePassthrough(),
            },
            prompt,
            model,
            new StringOutputParser(),
        ]);

        // Invoke the chain with a specific question.
        
        let question = req.body.question;

        const answer = await chain.invoke(question);
        

        // console.log(answerText);

        fs.unlinkSync(pdfFilePath);
        res.json({ answer });

        } catch (err) {
            console.error(err);
            res.status(500).send('Internal Server Error');
        }
});

app.post('/summarize-csv', upload.single('csv'),async (req, res) => {
    try {
        const model = new ChatOpenAI({
            modelName: "gpt-3.5-turbo"
        })
        if (!req.file) {
            return res.status(400).send('No CSV file uploaded.');
        }
        
        const csvFilePath = path.join(__dirname, 'uploads', req.file.originalname);
        fs.writeFileSync(csvFilePath, req.file.buffer);

        // Step-1 Load PDF

        // provide the pdf file path here
        const loader = new CSVLoader(csvFilePath);
        // const loader = new PDFLoader(pdfFilePath);
        const docs = await loader.load();
        
        // Step-2 Split the pdf into chunks

        //splitter function
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        // created chunks from pdf
        const splittedDocs = await splitter.splitDocuments(docs);
        // we will use OpenAI's embedding models
        const embeddings = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY// In Node.js defaults to process.env.OPENAI_API_KEY
            // batchSize: 512, // Default value if omitted is 512. Max is 2048
        });
        // Create a vector store from the documents.
        const vectorStore = await FaissStore.fromDocuments(
            splittedDocs,
            embeddings
        );

        // Initialize a retriever wrapper around the vector store.
        const vectorStoreRetriever = vectorStore.asRetriever();

        // Create a system & human prompt for the chat model.
        const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    {context}`;
        const messages = [
            SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.fromTemplate("{question}"),
        ];
        const prompt = ChatPromptTemplate.fromMessages(messages);

        // Construct the runnable chain.
        const chain = RunnableSequence.from([
            {
                context: vectorStoreRetriever.pipe(formatDocumentsAsString),
                question: new RunnablePassthrough(),
            },
            prompt,
            model,
            new StringOutputParser(),
        ]);

        // Invoke the chain with a specific question.
        
        let question = req.body.question;

        const answer = await chain.invoke(question);
        

        // console.log(answerText);

        fs.unlinkSync(csvFilePath);
        res.json({ answer });

        } catch (err) {
            console.error(err);
            res.status(500).send('Internal Server Error');
        }
});

const rapid = new RapidAPI(process.env.RAPIDAPI_API_KEY);

app.post('/summarize-yt', async (req, res) => {
    try {
        const youtubeUrl = req.body.url;

        // Validate the YouTube URL
        if (!isValidYouTubeUrl(youtubeUrl)) {
            return res.status(400).send('Invalid YouTube URL.');
        }

        // Create a URL object
        const url = new URL(youtubeUrl);

        // Get the video ID from the 'v' parameter
        const videoId = url.searchParams.get("v");

        console.log(videoId);

        // Make a request using the RapidAPI SDK
        const response = await rapid.call('YouTubeToMp3', 'dl', {
            'id': videoId,
        });

        // Handle the 'on' event to download the MP3 file
        response.on('on', (data) => {
            // Assuming 'data' contains the MP3 file content
            // You may need to adjust this based on the actual API response
            const fileName = `downloaded_${videoId}.mp3`;

            // Save the file to your server
            // Note: Ensure you have appropriate error handling here
            fs.writeFileSync(fileName, data);

            // Provide the file for download or send a success response
            res.download(fileName, (err) => {
                if (err) {
                    console.error('Download error:', err);
                    res.status(500).send('Internal Server Error');
                } else {
                    // Clean up: remove the temporary file
                    fs.unlinkSync(fileName);
                }
            });
        });

        // Continue with the rest of your code...
    } catch (err) {
        console.error(err);
        res.status(500).send('Internal Server Error');
    }
});


// Helper function to validate YouTube URL
function isValidYouTubeUrl(url) {
    // Use a regular expression or other validation method to check if the URL is a valid YouTube URL
    // For simplicity, here's a basic regex that checks if the URL contains "youtube.com" and "v=" in the query parameters
    const youtubeRegex = /(?:https?:\/\/)?(?:www\.)?youtube\.com\/.*[?&]v=/;
    return youtubeRegex.test(url);
}

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
