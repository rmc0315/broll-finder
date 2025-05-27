# 🎬 AI-Powered B-Roll Finder

An intelligent tool that analyzes video scripts and automatically finds relevant B-roll footage from stock video sites. Powered by OpenAI for smart scene understanding and keyword extraction.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![AI-Powered](https://img.shields.io/badge/AI-Powered-purple.svg)

## ✨ Features

- **🤖 AI-Powered Analysis**: Uses OpenAI GPT to understand scene context and extract relevant visual keywords
- **🎯 Smart Search**: Generates multiple search queries per scene for better coverage
- **📺 Quality Filtering**: Prioritizes HD/4K footage with ideal B-roll duration (10-30s)
- **⏱️ Duration Matching**: Estimates scene duration and finds matching footage
- **💾 Smart Caching**: Caches search results for faster repeated analysis
- **📊 Beautiful Reports**: Generates interactive HTML reports with video previews
- **🔍 Multi-Source**: Searches Pexels and Pixabay (more sources coming soon)

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/broll-finder.git
cd broll-finder
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Get API keys
- **Pexels**: Free at https://www.pexels.com/api/
- **Pixabay**: Free at https://pixabay.com/api/docs/
- **OpenAI** (optional but recommended): https://platform.openai.com/api-keys

### 4. Run the script
```bash
python main.py
```

## 📝 Usage

### Interactive Mode (Recommended)
```bash
python main.py
```
Then paste your script when prompted and type 'END' when finished.

### File Mode
```bash
python main.py your_script.txt
```

### Demo Mode
```bash
python main.py --demo
```

## 📋 Script Format Example

```
[Opening Scene – Soft ambient music, deep ocean fade-in]
Narrator:
Beneath the ocean's shimmering surface... floats a creature so surreal, 
it could be mistaken for something from another world.

Slow-motion footage of a glowing jellyfish drifting through turquoise water. 
Bubbles rise, sunlight flickers above.

[Scene 2 – Close-up of jellyfish tentacles]
Narrator:
With no brain, no heart, and no bones... jellyfish are living ghosts of the sea.
```

## 🎯 How It Works

1. **Script Parsing**: Intelligently separates scenes, narrator text, and visual directions
2. **AI Analysis**: OpenAI analyzes each scene to extract visual keywords and generate search queries
3. **Multi-Source Search**: Searches multiple stock footage sites simultaneously
4. **Quality Scoring**: Each video is scored based on:
   - Resolution (4K > HD > SD)
   - Duration (10-30s ideal)
   - Scene duration matching
   - Source reliability
5. **Smart Caching**: Results are cached for 7 days for faster re-analysis
6. **Report Generation**: Creates beautiful HTML reports with previews and quality badges

## 📊 Output

The tool generates:
- **JSON file**: Machine-readable results with all metadata
- **HTML report**: Visual report with video previews, quality badges, and scene information

### Sample Output
```
✅ Analysis Complete!
Found 6 scenes in your script
Generated 60 B-roll suggestions
✨ 6 scenes enhanced with AI
📺 45 videos in Full HD or higher
💾 Cache contains 24 saved searches
```

## 🔧 Configuration

### Environment Variables
```bash
export PEXELS_API_KEY="your-key-here"
export PIXABAY_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

### Or use the interactive setup
The script will prompt for API keys on first run and save them locally.

## 🎨 Example Results

### Keywords Extracted by AI:
- Scene 1: `["ocean", "jellyfish", "underwater", "glowing", "bubbles", "sunlight"]`
- Scene 2: `["jellyfish", "translucent", "tentacles", "underwater", "elegance"]`

### Quality Scoring:
- 🏆 **HD+** (Score 6+): 4K resolution, ideal duration
- 🥈 **HD** (Score 4+): Full HD, good duration
- 🥉 **SD** (Score 2+): Lower resolution but usable

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:

- Add more stock footage sources (Unsplash, Coverr, Videvo)
- Implement style preferences (color, mood, camera movement)
- Add batch processing for multiple scripts
- Create video preview montages
- Add export formats (EDL, CSV, XML)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Pexels](https://www.pexels.com/) for their excellent free stock footage API
- [Pixabay](https://pixabay.com/) for their comprehensive video library
- [OpenAI](https://openai.com/) for GPT-powered scene understanding
- [spaCy](https://spacy.io/) for fallback NLP processing

## ⚠️ Important Notes

- **API Keys**: Keep your API keys secure and never commit them to version control
- **Rate Limits**: Be mindful of API rate limits, especially for free tiers
- **Copyright**: Always verify licensing terms for any stock footage you use
- **Costs**: OpenAI API usage incurs small costs (~$0.001 per scene)

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Error**: The script auto-downloads required data, but if issues persist:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
   ```

2. **SpaCy Model Error**: 
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **API Key Issues**: Make sure keys are valid and have necessary permissions

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

Made with ❤️ for video creators everywhere