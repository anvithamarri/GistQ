from transformers import BartForConditionalGeneration, BartTokenizer
import re
from collections import Counter

def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("�", "")
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

class SmartSummarizer:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.max_input_tokens = 1024

    def generate_summary(self, text, level="summary"):
        text = clean_text(text)
        original_sentences = split_sentences(text)
        
        if not original_sentences:
            return ""
        
        unique_sentences = self._remove_duplicates(original_sentences)
        
        # Word limits based on level
        if level == "abstract":
            target_words = 60
        elif level == "article":
            target_words = 250
        else:  # summary
            target_words = 130

        # Check if text is too long for single pass
        full_text = " ".join(unique_sentences)
        tokens = self.tokenizer(full_text, return_tensors="pt", truncation=False)
        num_tokens = tokens['input_ids'].shape[1]
        
        if num_tokens > self.max_input_tokens:
            summary = self._hierarchical_summarize(unique_sentences, target_words)
        else:
            summary = self._direct_summarize(full_text, target_words)
        
        return summary

    def _direct_summarize(self, text, target_words):
        """Summarize text that fits within token limit"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        # Set token limits to match word targets closely
        # Use 1 word ≈ 1.3-1.5 tokens ratio
        min_tokens = int(target_words * 0.9 * 1.3)
        max_tokens = int(target_words * 1.3 * 1.5)
        
        summary_ids = self.model.generate(
            **inputs,
            max_length=max_tokens,
            min_length=min_tokens,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def _hierarchical_summarize(self, sentences, target_words):
        """
        Multi-stage summarization for long articles
        """
        total_sentences = len(sentences)
        
        # Create balanced chunks
        chunks = self._create_balanced_chunks(sentences)
        
        print(f"Processing {total_sentences} sentences in {len(chunks)} chunks")
        
        # Stage 1: Summarize each chunk generously
        print("Stage 1: Chunk summarization...")
        chunk_summaries = []
        
        # Allocate more words per chunk to ensure rich intermediate summaries
        words_per_chunk = max(80, int(target_words * 0.7))
        
        for i, chunk in enumerate(chunks):
            chunk_text = " ".join(chunk)
            chunk_summary = self._direct_summarize(chunk_text, words_per_chunk)
            chunk_summaries.append(chunk_summary)
            print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk_summary.split())} words")
        
        # Stage 2: Combine all chunk summaries
        combined_text = " ".join(chunk_summaries)
        combined_words = len(combined_text.split())
        print(f"\nStage 2: Combined chunks = {combined_words} words")
        
        # Stage 3: Final summarization to target length
        print(f"Stage 3: Final compression to {target_words} words...")
        final_summary = self._direct_summarize(combined_text, target_words)
        
        final_words = len(final_summary.split())
        coverage = self._verify_coverage(sentences, final_summary)
        
        print(f"\n✓ Final output: {final_words} words | Coverage: {coverage:.1f}%\n")
        
        return final_summary

    def _create_balanced_chunks(self, sentences):
        """Create chunks ensuring coverage of beginning, middle, and end"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_limit = 900  # Safe limit under 1024
        
        total = len(sentences)
        # Force breaks at 33% and 66% points
        break_points = {total // 3, 2 * total // 3}
        
        for idx, sent in enumerate(sentences):
            sent_tokens = self.tokenizer(sent, return_tensors="pt", truncation=False)
            sent_token_count = sent_tokens['input_ids'].shape[1]
            
            # Force chunk break at major sections
            if idx in break_points and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            # Check token limit
            if current_tokens + sent_token_count > chunk_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [sent]
                current_tokens = sent_token_count
            else:
                current_chunk.append(sent)
                current_tokens += sent_token_count
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _verify_coverage(self, original_sentences, summary):
        """Calculate content coverage percentage"""
        original_text = " ".join(original_sentences).lower()
        original_concepts = set(re.findall(r'\b[a-z]{5,}\b', original_text))
        
        summary_concepts = set(re.findall(r'\b[a-z]{5,}\b', summary.lower()))
        
        if not original_concepts:
            return 100.0
        
        coverage = len(original_concepts & summary_concepts) / len(original_concepts) * 100
        return coverage

    def _remove_duplicates(self, sentences):
        """Remove duplicate sentences"""
        seen = set()
        unique = []
        
        for sent in sentences:
            normalized = sent.strip().rstrip(".!?").lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(sent)
        
        return unique


# Flask-compatible entry point
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = SmartSummarizer()
    return _summarizer

def generate_summary(text, level="summary"):
    return get_summarizer().generate_summary(text, level)


