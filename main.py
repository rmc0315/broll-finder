import re
import requests
from typing import List, Dict, Tuple
import json
from collections import Counter
import nltk
import spacy
import sys
import os
import hashlib
import time
from datetime import datetime
from openai import OpenAI
import csv

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
        except LookupError:
            print(f"Downloading required NLTK data: {data}...")
            nltk.download(data, quiet=True)

# Download NLTK data on import
ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class SearchCache:
    """Simple caching system for search results"""
    def __init__(self, cache_file='search_cache.json'):
        self.cache_file = cache_file
        self.cache = self.load_cache()
    
    def load_cache(self) -> Dict:
        """Load cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get_cached_results(self, query: str, source: str) -> List[Dict]:
        """Check if we've searched this before"""
        cache_key = f"{source}:{hashlib.md5(query.encode()).hexdigest()}"
        
        if cache_key in self.cache:
            # Check if cache is fresh (< 7 days old)
            if time.time() - self.cache[cache_key]['timestamp'] < 604800:
                return self.cache[cache_key]['results']
        return None
    
    def store_results(self, query: str, source: str, results: List[Dict]):
        """Store search results in cache"""
        cache_key = f"{source}:{hashlib.md5(query.encode()).hexdigest()}"
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'query': query,
            'results': results
        }
        self.save_cache()


class BRollFinder:
    def __init__(self, pexels_api_key: str = None, pixabay_api_key: str = None, 
                 openai_api_key: str = None, unsplash_api_key: str = None, 
                 vimeo_api_key: str = None):
        """
        Initialize the B-Roll finder with API keys for stock footage sites.
        
        Args:
            pexels_api_key: API key for Pexels (free stock videos)
            pixabay_api_key: API key for Pixabay (free stock videos)
            openai_api_key: API key for OpenAI (for intelligent keyword extraction)
            unsplash_api_key: API key for Unsplash (requires special access for videos)
            vimeo_api_key: API key for Vimeo (stock footage)
        """
        self.pexels_api_key = pexels_api_key
        self.pixabay_api_key = pixabay_api_key
        self.openai_api_key = openai_api_key
        self.unsplash_api_key = unsplash_api_key
        self.vimeo_api_key = vimeo_api_key
        
        # Initialize cache
        self.cache = SearchCache()
        
        # Count available sources
        available_sources = []
        if self.pexels_api_key:
            available_sources.append('Pexels')
        if self.pixabay_api_key:
            available_sources.append('Pixabay')
        if self.unsplash_api_key:
            available_sources.append('Unsplash')
        if self.vimeo_api_key:
            available_sources.append('Vimeo')
        
        print(f"✅ Initialized with {len(available_sources)} video sources: {', '.join(available_sources)}")
        
        # Initialize OpenAI client if API key provided
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            print("✅ OpenAI API initialized for intelligent keyword extraction")
        else:
            self.openai_client = None
            print("ℹ️  No OpenAI API key provided. Using basic keyword extraction.")
        
        # Load spaCy model for fallback NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_keywords_with_ai(self, text: str, scene_number: int) -> Tuple[List[str], List[str]]:
        """
        Use OpenAI to intelligently extract keywords and generate search queries.
        
        Args:
            text: Scene text
            scene_number: Scene number for context
            
        Returns:
            Tuple of (keywords, search_queries)
        """
        try:
            prompt = f"""
            Analyze this video script scene and provide keywords and search queries for finding relevant B-roll footage.
            
            Scene {scene_number}:
            {text}
            
            Provide your response in JSON format with two arrays:
            1. "keywords": 5-7 specific, visual keywords (things that can be filmed)
            2. "search_queries": 3-4 search queries for stock footage sites (2-4 words each)
            
            Focus on concrete, visual elements. Avoid abstract concepts or narrative terms.
            
            Example response format:
            {{
                "keywords": ["ocean", "jellyfish", "underwater", "glowing", "translucent"],
                "search_queries": ["jellyfish underwater", "ocean jellyfish swimming", "glowing sea creature", "translucent jellyfish"]
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a video production assistant helping find relevant B-roll footage."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            keywords = result.get('keywords', [])
            search_queries = result.get('search_queries', [])
            
            return keywords, search_queries
            
        except Exception as e:
            print(f"AI extraction failed for scene {scene_number}: {e}")
            # Fall back to basic extraction
            return None, None
    
    def filter_high_quality_videos(self, videos: List[Dict], scene_duration: float = None) -> List[Dict]:
        """
        Filter and score videos based on quality criteria.
        
        Args:
            videos: List of video results
            scene_duration: Estimated duration of the scene (optional)
            
        Returns:
            Sorted list of videos by quality score
        """
        for video in videos:
            score = 0
            
            # Resolution scoring (0-3 points)
            if video['width'] >= 3840:  # 4K
                score += 3
            elif video['width'] >= 1920:  # Full HD
                score += 2.5
            elif video['width'] >= 1280:  # HD
                score += 1.5
            else:
                score += 0.5
            
            # Duration scoring (0-2 points)
            # Ideal B-roll duration is 10-30 seconds
            duration = video.get('duration', 0)
            if 10 <= duration <= 30:
                score += 2
            elif 5 <= duration <= 60:
                score += 1
            elif duration > 60:
                score += 0.5
            
            # Scene duration matching (0-1 point)
            if scene_duration and duration:
                duration_diff = abs(duration - scene_duration)
                if duration_diff <= 5:
                    score += 1
                elif duration_diff <= 10:
                    score += 0.5
            
            # Source reliability (0-1 point)
            # Pexels generally has higher quality curation
            if video['source'] == 'Pexels':
                score += 1
            elif video['source'] == 'Pixabay':
                score += 0.7
            
            # Tag relevance (0-1 point) - for Pixabay videos
            if 'tags' in video and len(video.get('tags', [])) > 5:
                score += 0.5
            
            video['quality_score'] = score
        
        # Sort by quality score
        return sorted(videos, key=lambda x: x.get('quality_score', 0), reverse=True)
    
    def estimate_scene_duration(self, text: str) -> float:
        """
        Estimate how long the narration takes to read.
        
        Args:
            text: Narrator text
            
        Returns:
            Estimated duration in seconds
        """
        # Average speaking rate: 150 words per minute
        # Account for pauses and dramatic effect
        word_count = len(text.split())
        
        # Add extra time for punctuation pauses
        pause_chars = text.count('.') + text.count('!') + text.count('?') + text.count('...')
        
        # Base duration
        duration = (word_count / 150) * 60  # Convert to seconds
        
        # Add pause time (0.5 seconds per punctuation)
        duration += pause_chars * 0.5
        
        return duration
    
    def parse_script(self, script: str) -> List[Dict]:
        """
        Parse the video script into scenes/segments with timestamps if available.
        
        Args:
            script: The video script text
            
        Returns:
            List of dictionaries containing scene information
        """
        scenes = []
        
        # Clean up the script - remove title and metadata lines
        lines = script.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip title lines, empty lines, and metadata
            if line.strip() and not line.strip().startswith('🎬') and not line.strip().startswith('(') and not line.strip().endswith(')'):
                cleaned_lines.append(line)
        
        cleaned_script = '\n'.join(cleaned_lines)
        
        # Look for scene markers like [Scene X], [Opening Scene], etc.
        scene_pattern = r'\[((?:Opening |Closing |Scene \d+)[^]]*)\]'
        scene_splits = re.split(scene_pattern, cleaned_script)
        
        if len(scene_splits) > 1:
            # Script has scene markers
            current_scene_name = None
            for i, segment in enumerate(scene_splits):
                if i % 2 == 1:  # This is a scene name
                    current_scene_name = segment
                elif segment.strip():  # This is scene content
                    # Combine narrator text and visual directions
                    narrator_text = ""
                    visual_text = ""
                    
                    # Extract narrator lines (usually after "Narrator:")
                    narrator_match = re.search(r'Narrator:\s*([^[]*?)(?=\n\n|\Z)', segment, re.DOTALL)
                    if narrator_match:
                        narrator_text = narrator_match.group(1).strip()
                    
                    # Everything else is visual direction
                    visual_text = re.sub(r'Narrator:\s*[^[]*?(?=\n\n|\Z)', '', segment).strip()
                    
                    # Combine both for better context
                    combined_text = f"{narrator_text} {visual_text}".strip()
                    
                    if combined_text:
                        scenes.append({
                            'scene_name': current_scene_name,
                            'narrator_text': narrator_text,
                            'visual_direction': visual_text,
                            'text': combined_text,
                            'scene_number': len(scenes) + 1
                        })
        else:
            # No clear scene markers, try to identify scenes by structure
            # Look for patterns like "Narrator:" or visual cues in italics
            paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
            
            current_scene = {'narrator_text': '', 'visual_direction': '', 'text': ''}
            
            for para in paragraphs:
                if 'Narrator:' in para:
                    # This paragraph contains narration
                    if current_scene['text']:
                        # Save previous scene
                        scenes.append({
                            **current_scene,
                            'scene_number': len(scenes) + 1
                        })
                        current_scene = {'narrator_text': '', 'visual_direction': '', 'text': ''}
                    
                    # Extract narrator text
                    narrator_match = re.search(r'Narrator:\s*(.+)', para, re.DOTALL)
                    if narrator_match:
                        current_scene['narrator_text'] = narrator_match.group(1).strip()
                else:
                    # This is likely visual direction
                    current_scene['visual_direction'] += ' ' + para
                
                current_scene['text'] = f"{current_scene['narrator_text']} {current_scene['visual_direction']}".strip()
            
            # Don't forget the last scene
            if current_scene['text']:
                scenes.append({
                    **current_scene,
                    'scene_number': len(scenes) + 1
                })
        
        return scenes
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        Extract relevant keywords from text for searching B-roll (fallback method).
        
        Args:
            text: Text to analyze
            num_keywords: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        keywords = []
        text_lower = text.lower()
        
        if self.nlp:
            # Use spaCy for better keyword extraction
            doc = self.nlp(text_lower)
            
            # Extract different types of important words
            nouns = []
            verbs = []
            adjectives = []
            entities = []
            
            for token in doc:
                if token.pos_ == 'NOUN' and not token.is_stop and len(token.text) > 2:
                    nouns.append(token.text)
                elif token.pos_ == 'VERB' and not token.is_stop and len(token.text) > 3:
                    # Get the lemma form of verbs for better search results
                    verbs.append(token.lemma_)
                elif token.pos_ == 'ADJ' and not token.is_stop and len(token.text) > 3:
                    adjectives.append(token.text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'FAC']:
                    entities.append(ent.text.lower())
            
            # Combine and prioritize
            # Priority: Entities > Nouns > Adjectives > Verbs
            all_keywords = entities + nouns + adjectives + verbs
            
            # Remove duplicates while preserving order
            seen = set()
            for word in all_keywords:
                if word not in seen:
                    seen.add(word)
                    keywords.append(word)
                    if len(keywords) >= num_keywords:
                        break
        
        else:
            # Fallback to NLTK
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                # Basic stop words if NLTK data not available
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                             'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                             'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                             'does', 'did', 'will', 'would', 'could', 'should', 'may',
                             'might', 'must', 'shall', 'can', 'this', 'that', 'these',
                             'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                             'narrator', 'scene', 'shot', 'footage', 'image', 'text',
                             'fade', 'cut', 'pan', 'zoom', 'angle', 'frame'}
            
            try:
                word_tokens = word_tokenize(text_lower)
            except LookupError:
                # Basic tokenization
                word_tokens = re.findall(r'\b\w+\b', text_lower)
            
            # Filter words
            filtered_words = []
            for w in word_tokens:
                if (w.isalnum() and 
                    w not in stop_words and 
                    len(w) > 2 and 
                    not w.isdigit()):
                    filtered_words.append(w)
            
            # Count frequency and get most common
            word_freq = Counter(filtered_words)
            keywords = [word for word, _ in word_freq.most_common(num_keywords * 2)][:num_keywords]
        
        # Ensure we return at least some generic keywords if none found
        if not keywords:
            # Extract any words that aren't stop words
            words = text_lower.split()
            keywords = [w for w in words if w not in {'the', 'a', 'an', 'and', 'or', 'is', 'was', 'are', 'were'}][:num_keywords]
        
        return keywords[:num_keywords]
    
    def identify_scene_type(self, text: str) -> str:
        """
        Identify the type of scene to better match B-roll.
        
        Args:
            text: Scene text
            
        Returns:
            Scene type (e.g., 'action', 'dialogue', 'description', etc.)
        """
        text_lower = text.lower()
        
        # Define patterns for different scene types
        scene_patterns = {
            'action': ['running', 'jumping', 'driving', 'flying', 'fighting', 'moving', 
                      'walking', 'climbing', 'swimming', 'dancing', 'playing', 'riding',
                      'throwing', 'catching', 'hitting', 'kicking', 'pushing', 'pulling'],
            'emotion': ['happy', 'sad', 'angry', 'excited', 'worried', 'calm', 'afraid',
                       'surprised', 'disgusted', 'love', 'hate', 'joy', 'fear', 'anxious',
                       'peaceful', 'stressed', 'relaxed', 'tense', 'crying', 'laughing'],
            'location': ['city', 'forest', 'beach', 'mountain', 'office', 'home', 'street',
                        'park', 'building', 'room', 'house', 'apartment', 'store', 'mall',
                        'restaurant', 'cafe', 'hospital', 'school', 'university', 'field'],
            'nature': ['tree', 'flower', 'animal', 'bird', 'water', 'sky', 'cloud', 'sun',
                      'moon', 'star', 'rain', 'snow', 'wind', 'ocean', 'river', 'lake',
                      'mountain', 'valley', 'desert', 'forest', 'jungle', 'garden'],
            'technology': ['computer', 'phone', 'tablet', 'screen', 'internet', 'software',
                          'hardware', 'device', 'machine', 'robot', 'ai', 'digital', 'tech',
                          'electronic', 'circuit', 'code', 'program', 'app', 'website'],
            'people': ['person', 'people', 'man', 'woman', 'child', 'children', 'family',
                      'friend', 'group', 'crowd', 'team', 'couple', 'individual', 'human'],
            'abstract': ['concept', 'idea', 'thought', 'dream', 'imagine', 'believe', 'think',
                        'feel', 'understand', 'know', 'learn', 'discover', 'create', 'design']
        }
        
        # Count matches for each scene type
        type_scores = {}
        for scene_type, keywords in scene_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[scene_type] = score
        
        # Return the scene type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        # Default to general if no specific type detected
        return 'general'
    
    def search_pexels(self, query: str, per_page: int = 5) -> List[Dict]:
        """
        Search Pexels for stock videos.
        
        Args:
            query: Search query
            per_page: Number of results per page
            
        Returns:
            List of video results
        """
        if not self.pexels_api_key:
            return []
        
        # Check cache first
        cached_results = self.cache.get_cached_results(query, 'Pexels')
        if cached_results:
            print(f"  📦 Using cached results for Pexels: '{query}'")
            return cached_results
        
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": self.pexels_api_key}
        params = {
            "query": query,
            "per_page": per_page,
            "orientation": "landscape"  # Usually better for B-roll
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                videos = []
                for video in data.get('videos', []):
                    videos.append({
                        'source': 'Pexels',
                        'id': video['id'],
                        'url': video['url'],
                        'thumbnail': video['image'],
                        'duration': video['duration'],
                        'width': video['width'],
                        'height': video['height'],
                        'preview': video['video_files'][0]['link'] if video['video_files'] else None
                    })
                
                # Cache the results
                self.cache.store_results(query, 'Pexels', videos)
                return videos
        except Exception as e:
            print(f"Error searching Pexels: {e}")
        
        return []
    
    def search_pixabay(self, query: str, per_page: int = 5) -> List[Dict]:
        """
        Search Pixabay for stock videos.
        
        Args:
            query: Search query
            per_page: Number of results per page
            
        Returns:
            List of video results
        """
        if not self.pixabay_api_key:
            return []
        
        # Check cache first
        cached_results = self.cache.get_cached_results(query, 'Pixabay')
        if cached_results:
            print(f"  📦 Using cached results for Pixabay: '{query}'")
            return cached_results
        
        url = "https://pixabay.com/api/videos/"
        params = {
            "key": self.pixabay_api_key,
            "q": query,
            "per_page": per_page,
            "video_type": "film"  # Better for B-roll than animation
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                videos = []
                for video in data.get('hits', []):
                    # Get the best available video size
                    video_sizes = video.get('videos', {})
                    video_url = None
                    width = 0
                    height = 0
                    
                    # Try to get medium or large size
                    if 'medium' in video_sizes:
                        video_url = video_sizes['medium']['url']
                        width = video_sizes['medium']['width']
                        height = video_sizes['medium']['height']
                    elif 'large' in video_sizes:
                        video_url = video_sizes['large']['url']
                        width = video_sizes['large']['width']
                        height = video_sizes['large']['height']
                    elif 'small' in video_sizes:
                        video_url = video_sizes['small']['url']
                        width = video_sizes['small']['width']
                        height = video_sizes['small']['height']
                    
                    videos.append({
                        'source': 'Pixabay',
                        'id': video['id'],
                        'url': video['pageURL'],
                        'thumbnail': f"https://i.vimeocdn.com/video/{video['picture_id']}_295x166.jpg" if 'picture_id' in video else video.get('userImageURL', ''),
                        'duration': video['duration'],
                        'width': width,
                        'height': height,
                        'preview': video_url,
                        'tags': video.get('tags', '').split(', ')
                    })
                
                # Cache the results
                self.cache.store_results(query, 'Pixabay', videos)
                return videos
        except Exception as e:
            print(f"Error searching Pixabay: {e}")
        
        return []
    
    def search_unsplash(self, query: str, per_page: int = 5) -> List[Dict]:
        """
        Search Unsplash for stock videos (if API key available).
        Note: Unsplash requires application approval for API access.
        
        Args:
            query: Search query
            per_page: Number of results
            
        Returns:
            List of video results
        """
        # Note: This is a placeholder for Unsplash integration
        # Unsplash video API requires special access
        return []
    
    def search_coverr(self, query: str) -> List[Dict]:
        """
        Search Coverr for free stock videos.
        Note: Coverr doesn't have a public API, but could be scraped or manually curated.
        
        Args:
            query: Search query
            
        Returns:
            List of video results
        """
        # Placeholder for future Coverr integration
        # Could maintain a curated list of Coverr videos by category
        return []
    
    def find_broll_for_script(self, script: str) -> List[Dict]:
        """
        Main function to find B-roll footage for entire script.
        
        Args:
            script: The video script
            
        Returns:
            List of scenes with matched B-roll suggestions
        """
        # Parse script into scenes
        scenes = self.parse_script(script)
        
        # Find B-roll for each scene
        results = []
        
        for scene in scenes:
            keywords = []
            queries = []
            
            # Try AI extraction first if available
            if self.openai_client:
                ai_keywords, ai_queries = self.extract_keywords_with_ai(
                    scene['text'], 
                    scene['scene_number']
                )
                if ai_keywords and ai_queries:
                    keywords = ai_keywords
                    queries = ai_queries
                    print(f"✨ Scene {scene['scene_number']}: Using AI-powered extraction")
            
            # Fall back to traditional extraction if AI fails or unavailable
            if not keywords:
                keywords = self.extract_keywords(scene['text'], num_keywords=7)
                scene_type = self.identify_scene_type(scene['text'])
                
                # Create basic queries
                if len(keywords) >= 3:
                    queries.append(' '.join(keywords[:3]))
                elif keywords:
                    queries.append(' '.join(keywords))
                
                if keywords and scene_type != 'general':
                    queries.append(f"{scene_type} {keywords[0]}")
                
                if len(keywords) >= 2:
                    queries.append(f"{keywords[0]} {keywords[1]}")
                
                print(f"📝 Scene {scene['scene_number']}: Using basic extraction")
            
            # Limit queries
            queries = queries[:4]
            
            # Search for videos
            videos = []
            seen_ids = set()
            
            # Estimate scene duration for better matching
            scene_duration = self.estimate_scene_duration(scene.get('narrator_text', scene['text']))
            
            # Try each query
            for query in queries:
                if self.pexels_api_key:
                    pexels_results = self.search_pexels(query, per_page=4)
                    for video in pexels_results:
                        video_id = f"{video['source']}_{video['id']}"
                        if video_id not in seen_ids:
                            seen_ids.add(video_id)
                            videos.append(video)
                
                if self.pixabay_api_key:
                    pixabay_results = self.search_pixabay(query, per_page=4)
                    for video in pixabay_results:
                        video_id = f"{video['source']}_{video['id']}"
                        if video_id not in seen_ids:
                            seen_ids.add(video_id)
                            videos.append(video)
                
                # Stop if we have enough videos
                if len(videos) >= 12:
                    break
            
            # Apply quality filtering and sorting
            videos = self.filter_high_quality_videos(videos, scene_duration)
            
            # Build result for this scene
            scene_result = {
                'scene_number': scene['scene_number'],
                'scene_name': scene.get('scene_name', f"Scene {scene['scene_number']}"),
                'text': scene['text'][:200] + '...' if len(scene['text']) > 200 else scene['text'],
                'full_text': scene['text'],
                'narrator_text': scene.get('narrator_text', ''),
                'visual_direction': scene.get('visual_direction', ''),
                'keywords': keywords,
                'scene_type': self.identify_scene_type(scene['text']),
                'search_queries': queries,
                'estimated_duration': round(scene_duration, 1),
                'broll_suggestions': videos[:10],  # Top 10 suggestions per scene
                'ai_powered': bool(self.openai_client and keywords)  # Track if AI was used
            }
            
            results.append(scene_result)
        
        return results
    
    def export_results(self, results: List[Dict], output_file: str = 'broll_suggestions.json'):
        """
        Export B-roll suggestions to a file.
        
        Args:
            results: Results from find_broll_for_script
            output_file: Output filename
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {output_file}")
    
    def generate_html_report(self, results: List[Dict], output_file: str = 'broll_report.html'):
        """
        Generate an HTML report with B-roll suggestions.
        
        Args:
            results: Results from find_broll_for_script
            output_file: Output HTML filename
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>B-Roll Suggestions Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .scene { border: 1px solid #ddd; margin: 20px 0; padding: 15px; }
                .scene-header { background: #f0f0f0; padding: 10px; margin: -15px -15px 15px; }
                .video-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
                .video-item { border: 1px solid #ccc; padding: 10px; text-align: center; }
                .video-item img { max-width: 100%; height: auto; }
                .keywords { color: #666; font-style: italic; }
                .ai-badge { 
                    background: #4CAF50; 
                    color: white; 
                    padding: 2px 8px; 
                    border-radius: 12px; 
                    font-size: 12px; 
                    margin-left: 10px;
                }
                .quality-badge {
                    position: absolute;
                    top: 5px;
                    right: 5px;
                    background: #2196F3;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 10px;
                }
                .duration-info {
                    background: #f5f5f5;
                    padding: 5px 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <h1>B-Roll Suggestions for Your Video Script</h1>
        """
        
        for result in results:
            ai_badge = '<span class="ai-badge">AI Enhanced</span>' if result.get('ai_powered') else ''
            html += f"""
            <div class="scene">
                <div class="scene-header">
                    <h2>Scene {result['scene_number']}: {result.get('scene_name', '')}{ai_badge}</h2>
                </div>
                {f'<p><strong>Narrator:</strong> <em>{result["narrator_text"]}</em></p>' if result.get('narrator_text') else ''}
                {f'<p><strong>Visual Direction:</strong> {result["visual_direction"]}</p>' if result.get('visual_direction') else ''}
                <div class="duration-info">
                    <strong>Estimated Scene Duration:</strong> {result.get('estimated_duration', 'N/A')} seconds
                </div>
                <p class="keywords"><strong>Keywords:</strong> {', '.join(result['keywords'])}</p>
                <p><strong>Scene Type:</strong> {result['scene_type']}</p>
                <p><strong>Search Queries Used:</strong> {' | '.join(result.get('search_queries', []))}</p>
                
                <h3>B-Roll Suggestions:</h3>
                <div class="video-grid">
            """
            
            for video in result['broll_suggestions']:
                # Determine quality badge
                quality = ""
                if video.get('quality_score', 0) >= 6:
                    quality = "HD+"
                elif video.get('quality_score', 0) >= 4:
                    quality = "HD"
                elif video.get('quality_score', 0) >= 2:
                    quality = "SD"
                
                html += f"""
                <div class="video-item" style="position: relative;">
                    {f'<span class="quality-badge">{quality}</span>' if quality else ''}
                    <img src="{video['thumbnail']}" alt="Video thumbnail">
                    <p>{video['source']} - {video['duration']}s</p>
                    <p>{video['width']}x{video['height']}</p>
                    <p style="font-size: 11px; color: #666;">Score: {video.get('quality_score', 0):.1f}</p>
                    <a href="{video['url']}" target="_blank">View Video</a>
                </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML report generated: {output_file}")
    
    def export_csv(self, results: List[Dict], output_file: str = 'broll_suggestions.csv'):
        """
        Export B-roll suggestions to CSV for easy sharing and editing.
        
        Args:
            results: Results from find_broll_for_script
            output_file: Output CSV filename
        """
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow([
                'Scene #', 'Scene Name', 'Keywords', 'Est. Duration (s)',
                'Video Source', 'Video URL', 'Video Duration (s)', 
                'Resolution', 'Quality Score', 'Tags'
            ])
            
            # Data rows
            for scene in results:
                scene_num = scene['scene_number']
                scene_name = scene.get('scene_name', f"Scene {scene_num}")
                keywords = ', '.join(scene['keywords'])
                est_duration = scene.get('estimated_duration', 'N/A')
                
                for video in scene['broll_suggestions'][:5]:  # Top 5 per scene
                    writer.writerow([
                        scene_num,
                        scene_name,
                        keywords,
                        est_duration,
                        video['source'],
                        video['url'],
                        video['duration'],
                        f"{video['width']}x{video['height']}",
                        f"{video.get('quality_score', 0):.1f}",
                        ', '.join(video.get('tags', [])[:5]) if 'tags' in video else ''
                    ])
        
        print(f"✅ CSV export saved to: {output_file}")
    
    def export_markdown(self, results: List[Dict], output_file: str = 'broll_report.md'):
        """
        Export as a Markdown document for easy sharing and documentation.
        
        Args:
            results: Results from find_broll_for_script
            output_file: Output markdown filename
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("# B-Roll Suggestions Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            total_scenes = len(results)
            total_suggestions = sum(len(r['broll_suggestions']) for r in results)
            ai_scenes = sum(1 for r in results if r.get('ai_powered'))
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Scenes**: {total_scenes}\n")
            f.write(f"- **Total B-Roll Suggestions**: {total_suggestions}\n")
            f.write(f"- **AI-Enhanced Scenes**: {ai_scenes}\n\n")
            
            # Scene details
            f.write("## Scene Breakdown\n\n")
            
            for scene in results:
                f.write(f"### Scene {scene['scene_number']}: {scene.get('scene_name', '')}\n\n")
                
                if scene.get('ai_powered'):
                    f.write("🤖 **AI-Enhanced Analysis**\n\n")
                
                f.write(f"**Estimated Duration**: {scene.get('estimated_duration', 'N/A')} seconds\n\n")
                f.write(f"**Keywords**: {', '.join(scene['keywords'])}\n\n")
                
                if scene.get('narrator_text'):
                    f.write(f"**Narration**: _{scene['narrator_text'][:150]}{'...' if len(scene['narrator_text']) > 150 else ''}_\n\n")
                
                f.write("**Top B-Roll Suggestions**:\n\n")
                
                for i, video in enumerate(scene['broll_suggestions'][:5], 1):
                    quality = self._get_quality_label(video.get('quality_score', 0))
                    f.write(f"{i}. **{video['source']}** - {quality}\n")
                    f.write(f"   - Duration: {video['duration']}s\n")
                    f.write(f"   - Resolution: {video['width']}x{video['height']}\n")
                    f.write(f"   - [View Video]({video['url']})\n")
                    if 'tags' in video and video['tags']:
                        f.write(f"   - Tags: {', '.join(video['tags'][:5])}\n")
                    f.write("\n")
                
                # Manual source suggestions
                f.write("**Additional Manual Sources**:\n")
                keywords_query = '+'.join(scene['keywords'][:3])
                f.write(f"- [Search Coverr](https://coverr.co/s?q={keywords_query})\n")
                f.write(f"- [Search Videvo](https://www.videvo.net/search/{keywords_query}/)\n")
                f.write(f"- [Search Mazwai](https://mazwai.com/search?q={keywords_query})\n\n")
                
                f.write("---\n\n")
        
        print(f"✅ Markdown report saved to: {output_file}")
    
    # Helper methods
    def _get_quality_label(self, score: float) -> str:
        """Get quality label based on score"""
        if score >= 6:
            return "⭐ Premium Quality (4K)"
        elif score >= 4:
            return "✨ High Quality (HD+)"
        elif score >= 2:
            return "✓ Standard Quality (HD)"
        else:
            return "Basic Quality"


def get_multiline_input(prompt: str) -> str:
    """
    Get multiline input from user. End input with 'END' on a new line.
    """
    print(prompt)
    print("(Type or paste your script, then type 'END' on a new line when finished)")
    print("-" * 50)
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    
    return '\n'.join(lines)


def load_or_input_api_keys():
    """
    Load API keys from environment or prompt user to input them.
    """
    import os
    
    # Try to load from environment variables first
    pexels_key = os.environ.get('PEXELS_API_KEY')
    pixabay_key = os.environ.get('PIXABAY_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    # If not in environment, check if keys file exists
    keys_file = 'api_keys.json'
    if os.path.exists(keys_file):
        try:
            with open(keys_file, 'r') as f:
                keys = json.load(f)
                pexels_key = pexels_key or keys.get('pexels')
                pixabay_key = pixabay_key or keys.get('pixabay')
                openai_key = openai_key or keys.get('openai')
        except:
            pass
    
    # If still no keys, prompt user
    if not pexels_key and not pixabay_key:
        print("\n=== API Key Setup ===")
        print("You need at least one API key to search for B-roll footage.")
        print("Both services offer free API keys:")
        print("- Pexels: https://www.pexels.com/api/")
        print("- Pixabay: https://pixabay.com/api/docs/")
        print("\nFor better results, you can also add:")
        print("- OpenAI: https://platform.openai.com/api-keys (for AI-powered extraction)")
        print("\nPress Enter to skip if you don't have a key yet.\n")
        
        pexels_key = input("Enter your Pexels API key (or press Enter to skip): ").strip()
        pixabay_key = input("Enter your Pixabay API key (or press Enter to skip): ").strip()
        openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        
        # Offer to save keys for future use
        if pexels_key or pixabay_key or openai_key:
            save = input("\nSave API keys for future use? (y/n): ").lower()
            if save == 'y':
                with open(keys_file, 'w') as f:
                    json.dump({
                        'pexels': pexels_key,
                        'pixabay': pixabay_key,
                        'openai': openai_key
                    }, f)
                print("API keys saved to api_keys.json")
    
    return pexels_key or None, pixabay_key or None, openai_key or None


def interactive_cli():
    """
    Interactive command-line interface for the B-Roll finder.
    """
    print("\n" + "="*60)
    print("🎬 B-ROLL FOOTAGE FINDER FOR VIDEO SCRIPTS 🎬")
    print("="*60)
    
    # Load or get API keys
    pexels_key, pixabay_key, openai_key = load_or_input_api_keys()
    
    if not pexels_key and not pixabay_key:
        print("\n⚠️  Warning: No API keys provided. The script will not be able to search for footage.")
        print("Please get at least one API key from the services mentioned above.")
        return
    
    # Initialize the finder
    finder = BRollFinder(
        pexels_api_key=pexels_key,
        pixabay_api_key=pixabay_key,
        openai_api_key=openai_key
    )
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        print("\nOptions:")
        print("1. Analyze a new video script")
        print("2. Load script from file")
        print("3. View last results")
        print("4. Update API keys")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            # Get script from user input
            script = get_multiline_input("\nPaste your video script below:")
            
            if script.strip():
                print("\n🔍 Analyzing script and searching for B-roll footage...")
                results = finder.find_broll_for_script(script)
                
                # Show summary
                print("\n✅ Analysis Complete!")
                print(f"Found {len(results)} scenes in your script")
                
                total_suggestions = sum(len(r['broll_suggestions']) for r in results)
                print(f"Generated {total_suggestions} B-roll suggestions")
                
                ai_scenes = sum(1 for r in results if r.get('ai_powered'))
                if ai_scenes > 0:
                    print(f"✨ {ai_scenes} scenes enhanced with AI")
                
                # Show quality stats
                all_videos = [v for r in results for v in r['broll_suggestions']]
                hd_count = sum(1 for v in all_videos if v.get('width', 0) >= 1920)
                print(f"📺 {hd_count} videos in Full HD or higher")
                
                # Show cache stats
                cache_size = len(finder.cache.cache)
                if cache_size > 0:
                    print(f"💾 Cache contains {cache_size} saved searches")
                
                # Export options
                print("\n📁 Export Options:")
                export_json = input("Export to JSON? (y/n): ").lower() == 'y'
                export_html = input("Generate HTML report? (y/n): ").lower() == 'y'
                export_csv = input("Export to CSV? (y/n): ").lower() == 'y'
                export_md = input("Generate Markdown report? (y/n): ").lower() == 'y'
                
                if export_json:
                    filename = input("JSON filename (default: broll_suggestions.json): ").strip()
                    finder.export_results(results, filename or 'broll_suggestions.json')
                
                if export_html:
                    filename = input("HTML filename (default: broll_report.html): ").strip()
                    finder.generate_html_report(results, filename or 'broll_report.html')
                
                if export_csv:
                    filename = input("CSV filename (default: broll_suggestions.csv): ").strip()
                    finder.export_csv(results, filename or 'broll_suggestions.csv')
                
                if export_md:
                    filename = input("Markdown filename (default: broll_report.md): ").strip()
                    finder.export_markdown(results, filename or 'broll_report.md')
                
                # Show preview
                print("\n📋 Preview of Results:")
                for result in results[:3]:  # Show first 3 scenes
                    print(f"\nScene {result['scene_number']}:")
                    print(f"  Text: {result['text'][:100]}...")
                    print(f"  Keywords: {', '.join(result['keywords'])}")
                    print(f"  B-roll found: {len(result['broll_suggestions'])} videos")
                    if result.get('ai_powered'):
                        print(f"  ✨ AI-enhanced")
            
        elif choice == '2':
            # Load script from file
            filename = input("Enter script filename: ").strip()
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    script = f.read()
                
                print(f"\n✅ Loaded script from {filename}")
                print("🔍 Analyzing script and searching for B-roll footage...")
                
                results = finder.find_broll_for_script(script)
                
                # Same export options as above
                print("\n✅ Analysis Complete!")
                print(f"Found {len(results)} scenes in your script")
                
                finder.export_results(results)
                finder.generate_html_report(results)
                
            except FileNotFoundError:
                print(f"❌ Error: File '{filename}' not found")
            except Exception as e:
                print(f"❌ Error loading file: {e}")
        
        elif choice == '3':
            # View last results
            try:
                with open('broll_suggestions.json', 'r') as f:
                    results = json.load(f)
                
                print("\n📊 Last Results Summary:")
                for result in results:
                    print(f"\nScene {result['scene_number']}:")
                    print(f"  Keywords: {', '.join(result['keywords'])}")
                    print(f"  B-roll found: {len(result['broll_suggestions'])} videos")
                    if result.get('ai_powered'):
                        print(f"  ✨ AI-enhanced")
                
                view_html = input("\nOpen HTML report in browser? (y/n): ").lower() == 'y'
                if view_html:
                    import webbrowser
                    webbrowser.open('broll_report.html')
                    
            except FileNotFoundError:
                print("❌ No previous results found. Analyze a script first!")
        
        elif choice == '4':
            # Update API keys
            print("\n🔑 Update API Keys")
            pexels_key = input("Enter new Pexels API key (or press Enter to keep current): ").strip()
            pixabay_key = input("Enter new Pixabay API key (or press Enter to keep current): ").strip()
            openai_key = input("Enter new OpenAI API key (or press Enter to keep current): ").strip()
            
            if pexels_key:
                finder.pexels_api_key = pexels_key
            if pixabay_key:
                finder.pixabay_api_key = pixabay_key
            if openai_key:
                finder.openai_api_key = openai_key
                finder.openai_client = OpenAI(api_key=openai_key)
            
            # Save updated keys
            with open('api_keys.json', 'w') as f:
                json.dump({
                    'pexels': finder.pexels_api_key,
                    'pixabay': finder.pixabay_api_key,
                    'openai': finder.openai_api_key
                }, f)
            print("✅ API keys updated!")
        
        elif choice == '5':
            print("\n👋 Thanks for using B-Roll Finder! Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-5.")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage:")
            print("  python broll_finder.py              # Interactive mode")
            print("  python broll_finder.py <script.txt> # Process script file")
            print("  python broll_finder.py --demo       # Run with demo script")
        
        elif sys.argv[1] == '--demo':
            # Demo mode with sample script
            sample_script = """
            [00:00] Welcome to our cooking tutorial! Today we're making homemade pasta from scratch.
            
            [00:15] First, let's gather our ingredients: flour, eggs, olive oil, and a pinch of salt.
            
            [00:30] Now we'll create a well in the flour and crack the eggs into the center.
            
            [00:45] Mix everything together until you have a smooth, elastic dough.
            
            [01:00] Roll out the dough and cut it into your favorite pasta shape!
            """
            
            # Use demo API keys (these are examples - replace with real ones)
            finder = BRollFinder()
            print("🎬 Running demo analysis...")
            print("\n⚠️  Note: This is a demo. Add API keys to get actual B-roll suggestions!")
            
            results = finder.find_broll_for_script(sample_script)
            finder.generate_html_report(results, 'demo_report.html')
            print("\n✅ Demo complete! Check demo_report.html")
        
        else:
            # File mode
            try:
                with open(sys.argv[1], 'r') as f:
                    script = f.read()
                
                pexels_key, pixabay_key, openai_key = load_or_input_api_keys()
                finder = BRollFinder(
                    pexels_api_key=pexels_key, 
                    pixabay_api_key=pixabay_key,
                    openai_api_key=openai_key
                )
                
                results = finder.find_broll_for_script(script)
                finder.export_results(results)
                finder.generate_html_report(results)
                print(f"✅ Processed {sys.argv[1]} successfully!")
                
            except FileNotFoundError:
                print(f"❌ Error: File '{sys.argv[1]}' not found")
    
    else:
        # Interactive mode
        interactive_cli()