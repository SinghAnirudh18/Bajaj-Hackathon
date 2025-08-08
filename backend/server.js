const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');

const app = express();
const port = 3000;

// Middleware
app.set('view engine', 'ejs');
app.use(cors());
app.use(express.static('public'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Multer storage configuration
const storage = multer.diskStorage({
    destination: async (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'Uploads');
        try {
            await fs.mkdir(uploadDir, { recursive: true });
            cb(null, uploadDir);
        } catch (err) {
            cb(err);
        }
    },
    filename: (req, file, cb) => {
        cb(null, `${Date.now()}-${file.originalname}`);
    }
});

const fileFilter = (req, file, cb) => {
    const allowedTypes = ['.pdf', '.txt', '.docx', '.doc', '.md'];
    const fileExtension = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(fileExtension)) {
        cb(null, true);
    } else {
        cb(new Error('Unsupported file type. Use PDF, TXT, DOCX, DOC, or MD.'), false);
    }
};

const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
    fileFilter: fileFilter
}).single('document');

// In-memory storage for documents
let documents = [];
let documentNames = [];
let documentFilenames = [];

// Route to render the upload interface
app.get('/', (req, res) => {
    res.render('index');
});

// Document upload endpoint
app.post('/upload', async (req, res) => {
    upload(req, res, async (err) => {
        try {
            if (err) {
                if (err.code === 'LIMIT_FILE_SIZE') {
                    return res.status(400).json({ message: 'File too large. Max 10MB.' });
                }
                return res.status(400).json({ message: err.message });
            }

            if (!req.file) {
                return res.status(400).json({ message: 'No file uploaded' });
            }

            // Read the content based on file type
            let text = '';
            const ext = path.extname(req.file.originalname).toLowerCase();

            if (ext === '.txt' || ext === '.md') {
                text = await fs.readFile(req.file.path, 'utf-8');
            } else if (ext === '.pdf') {
                const dataBuffer = await fs.readFile(req.file.path);
                const pdfData = await pdfParse(dataBuffer);
                text = pdfData.text;
            } else if (ext === '.docx' || ext === '.doc') {
                const result = await mammoth.extractRawText({ path: req.file.path });
                text = result.value;
            } else {
                await fs.unlink(req.file.path).catch(() => {});
                return res.status(400).json({ message: 'Unsupported file format.' });
            }

            documents.push(text);
            documentNames.push(req.file.originalname);
            documentFilenames.push(req.file.filename);

            res.json({ 
                message: 'File uploaded successfully!', 
                filename: req.file.filename,
                originalname: req.file.originalname
            });

        } catch (error) {
            console.error('Upload error:', error);
            if (req.file) {
                await fs.unlink(req.file.path).catch(() => {});
            }
            res.status(500).json({ message: `Error uploading file: ${error.message}` });
        }
    });
});

// Prompt submission endpoint
app.post('/submit-prompt', (req, res) => {
    const { filename, prompt } = req.body;
    if (!filename || !prompt) {
        return res.status(400).json({ message: 'Filename and prompt are required.' });
    }
    
    const index = documentFilenames.indexOf(filename);
    if (index === -1) {
        return res.status(400).json({ message: 'Document not found.' });
    }
    
    // Return redirect URL instead of redirecting directly
    res.json({ 
        redirectUrl: `/chat?filename=${encodeURIComponent(filename)}&prompt=${encodeURIComponent(prompt)}`
    });
});
// Query endpoint (placeholder)
app.post('/query', (req, res) => {
    const { filename, query } = req.body;
    if (!filename || !query) {
        return res.status(400).json({ message: 'Filename and query are required.' });
    }
    const index = documentFilenames.indexOf(filename);
    if (index === -1) {
        return res.status(400).json({ message: 'Document not found.' });
    }
    // Placeholder response; replace with actual query logic
    const response = `Response to "${query}" from ${documentNames[index]}`;
    res.json({ response });
});

// Route to render the chat interface
app.get('/chat', (req, res) => {
    const { filename, prompt } = req.query;
    res.render('chat', { filename: filename || '', prompt: prompt || '' });
});

// Start server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});