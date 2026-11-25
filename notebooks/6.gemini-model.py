import os
import json
from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai

# ==========================================
# 1. API Configuration
# ==========================================
# Replace 'YOUR_API_KEY' with your actual Gemini API key, 
# or set it as an environment variable: export GEMINI_API_KEY="..."
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBbKEsp8f1JIvaml_Zq0zrwzjNT8YeQpDQ")
genai.configure(api_key=API_KEY)

# Using gemini-2.5-flash as it is fast and excellent at strict formatting/JSON extraction
model = genai.GenerativeModel('gemini-2.5-flash')

app = Flask(__name__)

# ==========================================
# 2. Integrated HTML/CSS/JS Interface
# ==========================================
# A single-page, dark-themed UI with deep blue tones
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroMed NER | Gemini Edition</title>
    <!-- Tailwind CSS (v3 fallback via CDN) for rapid styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/@phosphor-icons/web"></script>
    <!-- Tesseract.js for in-browser OCR -->
    <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>
    
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'sans-serif'] },
                    colors: {
                        dark: '#0B0e14',
                        surface: '#131823',
                        highlight: 'rgba(255,255,255,0.03)',
                        borderglow: 'rgba(56, 189, 248, 0.1)',
                    },
                    animation: {
                        'gradient-x': 'gradient-x 8s ease infinite',
                        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    },
                    keyframes: {
                        'gradient-x': {
                            '0%, 100%': {
                                'background-size': '200% 200%',
                                'background-position': 'left center'
                            },
                            '50%': {
                                'background-size': '200% 200%',
                                'background-position': 'right center'
                            }
                        }
                    }
                }
            }
        }
    </script>

    <style>
        body {
            background-color: #0B0e14;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
            background-attachment: fixed;
            color: #e2e8f0;
            overflow-x: hidden;
        }

        .glass-panel {
            background: rgba(19, 24, 35, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0B0e14; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }

        /* Entity Tags */
        .entity-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 500;
            margin: 2px 0;
            transition: all 0.2s;
            cursor: pointer;
            position: relative;
        }
        
        .entity-tag:hover {
            transform: scale(1.05);
            z-index: 10;
        }

        .entity-medication {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.1);
        }

        .entity-disease {
            background: rgba(244, 63, 94, 0.15);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
            box-shadow: 0 0 10px rgba(244, 63, 94, 0.1);
        }
        
        .entity-procedure {
            background: rgba(56, 189, 248, 0.15);
            color: #7dd3fc;
            border: 1px solid rgba(56, 189, 248, 0.3);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.1);
        }

        .entity-label {
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
            font-weight: 700;
        }

        .btn-modern {
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        .btn-modern::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300%;
            height: 300%;
            background: rgba(255,255,255,0.1);
            transform: translate(-50%, -50%) rotate(45deg) translateY(100%);
            transition: transform 0.6s ease;
        }
        .btn-modern:hover::after {
            transform: translate(-50%, -50%) rotate(45deg) translateY(-100%);
        }
        
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 4px;
        }
        .typing-indicator span {
            width: 6px;
            height: 6px;
            background-color: currentColor;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out both;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body class="antialiased min-h-screen flex flex-col relative pb-20">

    <!-- Decorative glowing orbs -->
    <div class="fixed top-0 left-1/4 w-96 h-96 bg-indigo-600/20 rounded-full blur-[128px] pointer-events-none"></div>
    <div class="fixed bottom-0 right-1/4 w-96 h-96 bg-emerald-600/10 rounded-full blur-[128px] pointer-events-none"></div>

    <nav class="w-full border-b border-white/5 bg-surface/80 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="p-2 bg-gradient-to-br from-indigo-500 to-cyan-400 rounded-xl shadow-lg shadow-indigo-500/20">
                    <i class="ph ph-dna text-xl text-white"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-white">NeuroMed <span class="font-light text-slate-400">Gemini Edition</span></h1>
                </div>
            </div>
        </div>
    </nav>

    <main class="flex-grow w-full max-w-7xl mx-auto px-6 py-12 flex flex-col lg:flex-row gap-8 relative z-10">
        
        <!-- Left Panel: Input -->
        <section class="w-full lg:w-5/12 flex flex-col gap-6">
            <div class="glass-panel p-1 rounded-2xl">
                <div class="bg-surface rounded-xl p-6 h-full flex flex-col">
                    <h2 class="text-lg font-semibold flex items-center gap-2 mb-4">
                        <i class="ph ph-stethoscope text-indigo-400"></i> Clinical Notes
                    </h2>
                    
                    <textarea id="text-input" 
                        class="w-full flex-grow min-h-[300px] bg-dark/50 border border-slate-700/50 rounded-xl p-4 text-slate-300 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all font-mono text-sm resize-none"
                        placeholder="Enter patient history, abstract, or clinical notes here..."></textarea>
                    
                    <div class="mt-4 flex gap-3">
                        <button id="analyze-btn" class="btn-modern flex-grow bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 px-6 rounded-xl flex justify-center items-center gap-2 shadow-lg shadow-indigo-600/20">
                            <span id="btn-text"><i class="ph ph-sparkle"></i> Analyze Text</span>
                            <div class="typing-indicator" id="loading-indicator"><span></span><span></span><span></span></div>
                        </button>
                        <button id="clear-btn" class="bg-slate-800 hover:bg-slate-700 text-slate-300 p-3 rounded-xl transition-colors border border-slate-700" title="Clear Text">
                            <i class="ph ph-trash"></i>
                        </button>
                    </div>

                    <div class="mt-4">
                        <label id="drop-zone" for="file-upload" class="flex items-center justify-center p-4 border-2 border-dashed border-slate-700/50 rounded-xl hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all cursor-pointer group">
                            <div class="flex flex-col items-center gap-1 text-slate-500 group-hover:text-indigo-400 transition-colors">
                                <i class="ph ph-image text-2xl"></i>
                                <span class="text-xs font-medium">Upload .txt or Image for OCR</span>
                            </div>
                            <input id="file-upload" type="file" accept=".txt,.png,.jpg,.jpeg" class="hidden" />
                        </label>
                    </div>

                    <div class="mt-8">
                        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Quick Load Snippets</p>
                        <div class="flex flex-col gap-2">
                            <button class="sample-btn text-left text-sm text-slate-400 hover:text-indigo-400 bg-dark/30 hover:bg-dark/50 p-3 rounded-lg border border-slate-800/50 transition-all truncate">
                                "The patient presented with severe acute respiratory syndrome and was prescribed remdesivir."
                            </button>
                            <button class="sample-btn text-left text-sm text-slate-400 hover:text-indigo-400 bg-dark/30 hover:bg-dark/50 p-3 rounded-lg border border-slate-800/50 transition-all truncate">
                                "Aspirin effectively suppresses fever and relieves headaches caused by viral pathogens."
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Right Panel: Results -->
        <section class="w-full lg:w-7/12 flex flex-col gap-6">
            <div id="empty-state" class="glass-panel p-1 rounded-2xl h-full flex items-center justify-center min-h-[500px]">
                <div class="text-center p-8">
                    <div class="w-20 h-20 bg-dark rounded-full flex items-center justify-center mx-auto mb-6 border border-slate-800">
                        <i class="ph ph-brain text-4xl text-slate-600"></i>
                    </div>
                    <h3 class="text-xl font-semibold text-slate-300 mb-2">Awaiting Input</h3>
                    <p class="text-slate-500 max-w-sm mx-auto">Enter medical text on the left and click Analyze to benchmark with Gemini.</p>
                </div>
            </div>

            <div id="results-container" class="hidden flex flex-col gap-4">
                <!-- Gemini Output -->
                <div id="model-output-gemini" class="model-output glass-panel p-1 rounded-xl">
                    <div class="bg-surface rounded-lg p-5">
                        <div class="flex justify-between items-center mb-4 border-b border-slate-800 pb-3">
                            <h3 class="font-semibold text-white flex items-center gap-2">
                                <i class="ph ph-cpu text-emerald-400"></i> Gemini Model
                            </h3>
                            <span class="text-xs py-1 px-2 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Generative AI</span>
                        </div>
                        <div id="out-gemini" class="text-[15px] leading-8 text-slate-300"></div>
                        <div id="error-gemini" class="text-red-400 hidden mt-2"></div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script>
        const textInput = document.getElementById('text-input');
        const analyzeBtn = document.getElementById('analyze-btn');
        const clearBtn = document.getElementById('clear-btn');
        const btnText = document.getElementById('btn-text');
        const loadingIndicator = document.getElementById('loading-indicator');
        const emptyState = document.getElementById('empty-state');
        const resultsContainer = document.getElementById('results-container');
        const fileUpload = document.getElementById('file-upload');
        const dropZone = document.getElementById('drop-zone');

        // File upload & Drag/Drop logic with OCR
        async function handleFileUpload(file) {
            if (file.name.endsWith('.txt')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    textInput.value = e.target.result;
                };
                reader.readAsText(file);
            } else if (file.type.startsWith('image/')) {
                const origVal = textInput.value;
                const origPlaceholder = textInput.placeholder;
                
                textInput.value = '';
                textInput.placeholder = `Extracting text from image (${file.name})... Please wait! ⌛`;
                textInput.disabled = true;
                
                try {
                    const result = await Tesseract.recognize(file, 'eng');
                    textInput.value = result.data.text;
                } catch (err) {
                    alert('Error extracting text from image.');
                    textInput.value = origVal;
                } finally {
                    textInput.disabled = false;
                    textInput.placeholder = origPlaceholder;
                }
            } else {
                alert('Please upload a valid .txt or image file (.png, .jpg).');
            }
        }

        fileUpload.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-indigo-500', 'bg-indigo-500/10');
            dropZone.classList.remove('border-slate-700/50');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-indigo-500', 'bg-indigo-500/10');
            dropZone.classList.add('border-slate-700/50');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-indigo-500', 'bg-indigo-500/10');
            dropZone.classList.add('border-slate-700/50');
            if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
        });

        // Sample logic
        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                textInput.value = e.target.innerText.replace(/"/g, '').trim();
            });
        });

        clearBtn.addEventListener('click', () => {
            textInput.value = '';
            resultsContainer.classList.add('hidden');
            emptyState.classList.remove('hidden');
        });

        // Advanced Entity Processor and Renderer for Gemini JSON
        function highlightEntities(originalText, entitiesData) {
            let html = originalText;
            
            // Collect all entities with their types
            let allEntities = [];
            if (entitiesData.DISEASE) {
                entitiesData.DISEASE.forEach(e => allEntities.push({text: e, type: 'disease', icon: '<i class="ph-fill ph-virus"></i>'}));
            }
            if (entitiesData.MEDICATION) {
                entitiesData.MEDICATION.forEach(e => allEntities.push({text: e, type: 'medication', icon: '<i class="ph-fill ph-pill"></i>'}));
            }
            if (entitiesData.PROCEDURE) {
                entitiesData.PROCEDURE.forEach(e => allEntities.push({text: e, type: 'procedure', icon: '<i class="ph-fill ph-stethoscope"></i>'}));
            }
            
            // Sort by length descending to replace longer entities first
            allEntities.sort((a, b) => b.text.length - a.text.length);
            
            // Replace entities in text safely
            // Note: This is a simplistic highlighter. For production, regex word boundaries or case-insensitive replacer is better.
            allEntities.forEach(ent => {
                if (!ent.text) return;
                // Escape regex characters
                const escapedText = ent.text.replace(/[-[\]{}()*+?.,\\\\^$|#\s]/g, '\\$&');
                const regex = new RegExp(`\\\\b(${escapedText})\\\\b`, 'gi');
                
                const replacement = `<span class="entity-tag entity-${ent.type}">
                               <span>$1</span>
                               <span class="entity-label flex items-center gap-1">${ent.icon} ${ent.type}</span>
                           </span>`;
                           
                html = html.replace(regex, replacement);
            });
            
            return html.replace(/\\n/g, '<br/>');
        }

        analyzeBtn.addEventListener('click', async () => {
            const text = textInput.value.trim();
            if (!text) return;

            // UI Loading state
            analyzeBtn.disabled = true;
            btnText.style.display = 'none';
            loadingIndicator.style.display = 'flex';
            document.getElementById('error-gemini').classList.add('hidden');

            try {
                const response = await fetch('/api/ner', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) throw new Error("Server error");
                
                const data = await response.json();
                
                if (data.error) throw new Error(data.error);

                // Process and inject
                document.getElementById('out-gemini').innerHTML = highlightEntities(text, data.entities);

                emptyState.classList.add('hidden');
                resultsContainer.classList.remove('hidden');

            } catch (err) {
                document.getElementById('error-gemini').textContent = "Analysis failed: " + err.message;
                document.getElementById('error-gemini').classList.remove('hidden');
                emptyState.classList.add('hidden');
                resultsContainer.classList.remove('hidden');
                console.error(err);
            } finally {
                // Restore loading state
                analyzeBtn.disabled = false;
                btnText.style.display = 'inline-flex';
                loadingIndicator.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

# ==========================================
# 3. Routes & Logic
# ==========================================

@app.route('/')
def home():
    # Serve the HTML string directly
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/ner', methods=['POST'])
def perform_ner():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # The Strict Rules Prompt: Forcing the LLM to act as an NER tagger
    prompt = f"""
    You are an expert Medical Named Entity Recognition (NER) system. 
    Analyze the following clinical text and extract entities strictly into the following three categories:
    1. DISEASE (Conditions, symptoms, disorders)
    2. MEDICATION (Drugs, treatments, dosages)

    Rules:
    - Return ONLY valid JSON format.
    - Do not include markdown code blocks (like ```json).
    - Do not include any conversational text or explanations.
    - If a category has no entities, return an empty array for it.

    Text to analyze:
    "{text}"

    Expected Output Format:
    {{
        "DISEASE": ["entity1", "entity2"],
        "MEDICATION": ["entity3"],
        "PROCEDURE": []
    }}
    """

    try:
        response = model.generate_content(prompt)
        # Parse the string response into an actual JSON object
        raw_output = response.text.strip()
        
        # Clean up in case the LLM ignored instructions and added markdown ticks
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
            
        entities = json.loads(raw_output)
        return jsonify({"entities": entities})
        
    except json.JSONDecodeError:
        return jsonify({"error": "Gemini returned malformed JSON", "raw": response.text}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Medical NER Interface... Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)