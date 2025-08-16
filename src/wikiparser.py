"""
Wikidump Parser for Word Embeddings
Processes Wikipedia XML dumps and extracts clean text content
Optimized for Skip-gram, SPPMI-SVD, and GloVe algorithms
Utilises keyword filtering for targeted content extraction
"""

import bz2
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator, Optional
from collections import Counter
import wikitextparser as wtp
from bs4 import BeautifulSoup
import re


class WikiParser:
    """
    Memory-efficient Wikipedia parser optimized for word embedding training
    Outputs clean text corpus with vocabulary statistics
    Uses keyword filtering for targeted content extraction
    """
    
    def __init__(self, 
                 dump_path: Path, 
                 output_dir: str, 
                 logger: logging.Logger,
                 min_article_length: int = 100,
                 max_article_length: int = 50000,
                 batch_size: int = 1000,
                 keywords: Optional[list[str]] = None,
                 min_keyword_count: int = 5):
        """
        Initialize WikiParser
        
        Args:
            dump_path: Path to Wikipedia dump file (.xml.bz2)
            output_dir: Directory to save processed corpus
            logger: Logger instance
            min_article_length: Minimum article length in characters
            max_article_length: Maximum article length in characters  
            batch_size: Number of articles to process before flushing to disk
            keywords: List of keywords to filter articles by
            min_keyword_count: Minimum number of keyword instances required in content
        """
        self.dump_path = dump_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.min_article_length = min_article_length
        self.max_article_length = max_article_length
        self.batch_size = batch_size
        self.min_keyword_count = min_keyword_count
        
        # Default keywords if none provided
        self.keywords = keywords or [
            "science", "physics", "chemistry", "biology", "mathematics", "research", "scientific",
            "technology", "ai", "artificial intelligence", "machine learning", "blockchain", 
            "cryptocurrency", "programming", "computer", "software", "digital",
            "health", "medicine", "pandemic", "vaccine", "healthcare", "treatment", "medical",
            "disease", "virus", "bacteria", "infection", "drug", "therapy",
            "politics", "government", "election", "policy", "law", "political", "parliament", 
            "congress", "senate", "legislation", "propaganda"
        ]
        
        # Convert keywords to lowercase for case-insensitive matching
        self.keywords_lower = [kw.lower() for kw in self.keywords]
        
        # Extract base filename from dump path (remove all extensions)
        base_filename = self.dump_path.name
        # Remove .xml.bz2 or .bz2 extensions
        if base_filename.endswith('.xml.bz2'):
            base_filename = base_filename[:-8]  # Remove .xml.bz2
        elif base_filename.endswith('.bz2'):
            base_filename = base_filename[:-4]  # Remove .bz2
        elif base_filename.endswith('.xml'):
            base_filename = base_filename[:-4]  # Remove .xml
        
        # Output files with dynamic naming
        self.corpus_file = self.output_dir / f"{base_filename}-corpus.txt"
        self.vocab_file = self.output_dir / f"{base_filename}-vocab_stats.json"
        self.checkpoint_file = self.output_dir / f"{base_filename}-checkpoint.json"
        
        # Processing state
        self.articles_processed = 0
        self.articles_iterated = 0  # Total articles seen (including filtered out)
        self.articles_keyword_filtered = 0  # Articles filtered out by keyword criteria
        self.total_words = 0
        self.vocab_counter = Counter()
        self.current_batch = []
        
        # Text cleaning regex patterns
        self.whitespace_pattern = re.compile(r'\s+')
        self.punct_pattern = re.compile(r'[^\w\s]')
        
        # Log filtering configuration
        self.logger.info(f"Keyword filtering enabled with {len(self.keywords)} keywords")
        self.logger.info(f"Filtering criteria: title contains keyword OR content has >{self.min_keyword_count} keyword instances")
    
    def parse_dump(self) -> Generator[dict, None, None]:
        """
        Parse Wikipedia dump and yield progress statistics
        """
        self.logger.info(f"Starting to parse dump: {self.dump_path}")
        
        # Load checkpoint
        start_position = self.__load_checkpoint()
        
        # Open corpus file for appending
        corpus_mode = 'a' if start_position > 0 else 'w'
        
        try:
            with bz2.open(self.dump_path, 'rt', encoding='utf-8') as xml_file, \
                 open(self.corpus_file, corpus_mode, encoding='utf-8') as corpus_out:
                
                context = ET.iterparse(xml_file, events=("end",))
                current_position = 0
                
                for event, elem in context:
                    if elem.tag.endswith("page"):
                        current_position += 1
                        
                        # Skip to checkpoint position
                        if current_position <= start_position:
                            elem.clear()
                            continue
                        
                        # Count articles we actually process (after checkpoint)
                        self.articles_iterated += 1
                        
                        try:
                            clean_text = self.__process_page(elem)
                            if clean_text:
                                self.current_batch.append(clean_text)
                                self.articles_processed += 1
                                
                                # Process batch when full
                                if len(self.current_batch) >= self.batch_size:
                                    self.__flush_batch(corpus_out)
                                    self.__save_checkpoint(current_position)
                                    
                                    yield {
                                        "articles_processed": self.articles_processed,
                                        "articles_keyword_filtered": self.articles_keyword_filtered,
                                        "total_words": self.total_words,
                                        "vocab_size": len(self.vocab_counter),
                                        "position": current_position
                                    }
                            
                            # Log progress every 100 articles
                            if self.articles_iterated % 100 == 0:
                                self.logger.info(
                                    f"Processed {self.articles_processed}/{self.articles_iterated} "
                                    f"({self.articles_processed/self.articles_iterated * 100:.1f}%) articles, "
                                    f"Keyword filtered: {self.articles_keyword_filtered}"
                                )
                        
                        except Exception as e:
                            self.logger.error(f"Error processing page at position {current_position}: {e}")
                        
                        finally:
                            elem.clear()
                
                # Flush remaining articles
                if self.current_batch:
                    self.__flush_batch(corpus_out)
                    self.__save_checkpoint(current_position)
                
                # Save final vocabulary statistics
                self.__save_vocab_stats()
                
                yield {
                    "articles_processed": self.articles_processed,
                    "articles_keyword_filtered": self.articles_keyword_filtered,
                    "total_words": self.total_words,
                    "vocab_size": len(self.vocab_counter),
                    "position": current_position,
                    "completed": True
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing dump: {e}")
            raise
    
    def __process_page(self, elem) -> Optional[str]:
        """
        Extract and clean text from a Wikipedia page element
        Now includes keyword filtering
        
        Returns:
            Clean text string or None if page should be skipped
        """
        # Extract title
        title_elem = elem.find("./{*}title")
        if title_elem is None or title_elem.text is None:
            return None
        
        title = title_elem.text.strip()
        
        # Skip special pages
        if any(skip_pattern in title.lower() for skip_pattern in [
            "disambiguation", "list of", "category:", "template:", 
            "file:", "user:", "wikipedia:", "help:", "portal:"
        ]):
            return None
        
        # Extract text content
        text_elem = elem.find("./{*}revision/{*}text")
        if text_elem is None or text_elem.text is None:
            return None
        
        text = text_elem.text
        
        # Skip redirects
        if text.strip().lower().startswith("#redirect"):
            return None
        
        # KEYWORD FILTERING: Check if article should be processed
        if not self.__passes_keyword_filter(title, text):
            self.articles_keyword_filtered += 1
            return None
        
        # Clean the text (only if it passes keyword filter)
        clean_text = self.__clean_text(text)
        
        # Filter by length
        if (len(clean_text) < self.min_article_length or 
            len(clean_text) > self.max_article_length):
            return None
        
        return clean_text
    
    def __passes_keyword_filter(self, title: str, raw_text: str) -> bool:
        """
        Check if article passes keyword filtering criteria
        
        Args:
            title: Article title
            raw_text: Raw article content
            
        Returns:
            True if article should be processed, False otherwise
        """
        title_lower = title.lower()
        
        # Check condition 1: Title contains keyword
        for keyword in self.keywords_lower:
            if keyword in title_lower:
                return True
        
        # Check condition 2: Content has more than min_keyword_count instances of keywords
        content_lower = raw_text.lower()
        keyword_count = 0
        
        for keyword in self.keywords_lower:
            # Count occurrences of this keyword
            keyword_count += content_lower.count(keyword)
            
            # Early exit if we've already found enough
            if keyword_count > self.min_keyword_count:
                return True
        
        return False
    
    def __clean_text(self, raw_text: str) -> str:
        """
        Comprehensive text cleaning for word embeddings
        """
        try:
            # Parse wikitext
            parsed = wtp.parse(raw_text)
            
            # Extract plain text (removes templates, links, etc.)
            clean_text = parsed.plain_text()
            
            # Remove HTML tags
            clean_text = self.__strip_html(clean_text)
            
            # Normalize whitespace
            clean_text = self.whitespace_pattern.sub(' ', clean_text)
            
            # Convert to lowercase for embeddings
            clean_text = clean_text.lower()
            
            # Remove excessive punctuation but keep sentence structure
            # Keep periods, commas, question marks, exclamation marks
            clean_text = re.sub(r'[^\w\s.,!?]', ' ', clean_text)
            
            # Clean up multiple spaces again
            clean_text = self.whitespace_pattern.sub(' ', clean_text)
            
            return clean_text.strip()
            
        except Exception as e:
            self.logger.warning(f"Error cleaning text: {e}")
            # Fallback cleaning
            return ' '.join(raw_text.split())
    
    def __strip_html(self, text: str) -> str:
        """Remove HTML tags"""
        try:
            return BeautifulSoup(text, "lxml").get_text()
        except Exception:
            # Fallback regex cleaning
            return re.sub(r'<[^>]+>', '', text)
    
    def __flush_batch(self, corpus_file) -> None:
        """Write current batch to corpus file and update vocabulary"""
        if not self.current_batch:
            return
        
        for text in self.current_batch:
            # Write to corpus (one article per line)
            corpus_file.write(text + '\n')
            
            # Update vocabulary statistics
            words = text.split()
            self.total_words += len(words)
            self.vocab_counter.update(words)
        
        # Flush to disk
        corpus_file.flush()
        
        self.logger.info(
            f"Flushed batch: {len(self.current_batch)} articles, "
            f"Total: {self.articles_processed} articles, "
            f"{self.total_words} words, "
            f"{len(self.vocab_counter)} unique words"
        )
        
        # Reset batch
        self.current_batch = []
    
    def __save_vocab_stats(self) -> None:
        """Save vocabulary statistics to JSON file"""
        vocab_stats = {
            "total_words": self.total_words,
            "unique_words": len(self.vocab_counter),
            "articles_processed": self.articles_processed,
            "articles_keyword_filtered": self.articles_keyword_filtered,
            "filter_rate": f"{self.articles_keyword_filtered/max(1, self.articles_iterated)*100:.1f}%",
            "keywords_used": self.keywords,
            "min_keyword_count": self.min_keyword_count,
            "top_words": dict(self.vocab_counter.most_common(1000)),
            "word_frequencies": dict(self.vocab_counter)
        }
        
        try:
            with open(self.vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab_stats, f, indent=2)
            
            self.logger.info(f"Saved vocabulary statistics to {self.vocab_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving vocabulary stats: {e}")
    
    def __save_checkpoint(self, position: int) -> None:
        """Save processing checkpoint"""
        checkpoint_data = {
            "position": position,
            "articles_processed": self.articles_processed,
            "articles_iterated": self.articles_iterated,
            "articles_keyword_filtered": self.articles_keyword_filtered,
            "total_words": self.total_words,
            "vocab_size": len(self.vocab_counter),
            "dump_file": str(self.dump_path),
            "keywords": self.keywords,
            "min_keyword_count": self.min_keyword_count
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def __load_checkpoint(self) -> int:
        """Load checkpoint and restore state"""
        if not self.checkpoint_file.exists():
            return 0
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Verify same dump file and filtering settings
            if (checkpoint_data.get("dump_file") != str(self.dump_path) or
                checkpoint_data.get("keywords") != self.keywords or
                checkpoint_data.get("min_keyword_count") != self.min_keyword_count):
                self.logger.warning("Checkpoint is for different configuration, starting fresh")
                return 0
            
            # Restore state
            self.articles_processed = checkpoint_data.get("articles_processed", 0)
            self.articles_iterated = checkpoint_data.get("articles_iterated", 0)
            self.articles_keyword_filtered = checkpoint_data.get("articles_keyword_filtered", 0)
            self.total_words = checkpoint_data.get("total_words", 0)
            
            # Load existing vocabulary if resuming
            if self.vocab_file.exists():
                with open(self.vocab_file, 'r') as f:
                    vocab_data = json.load(f)
                    self.vocab_counter = Counter(vocab_data.get("word_frequencies", {}))
            
            return checkpoint_data.get("position", 0)
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return 0
    
    def get_corpus_stats(self) -> dict:
        """Get current corpus statistics"""
        total_filtered = self.articles_iterated - self.articles_processed
        return {
            "articles_iterated": self.articles_iterated,
            "articles_processed": self.articles_processed,
            "articles_filtered_out_total": total_filtered,
            "articles_keyword_filtered": self.articles_keyword_filtered,
            "articles_other_filters": total_filtered - self.articles_keyword_filtered,
            "keyword_filter_rate": f"{self.articles_keyword_filtered/max(1, self.articles_iterated)*100:.1f}%",
            "total_filter_rate": f"{total_filtered/max(1, self.articles_iterated)*100:.1f}%",
            "total_words": self.total_words,
            "unique_words": len(self.vocab_counter),
            "corpus_file_size": self.corpus_file.stat().st_size if self.corpus_file.exists() else 0
        }


# Example usage for word embeddings with keyword filtering
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logger = logging.getLogger("wikiparser")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler('wikiparser.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.propagate = False
    
    # Parse dump with keyword filtering
    dump_path = Path("data/WikiDump-2019.bz2")
    if dump_path.exists():
        # Custom keywords (optional - defaults are provided)
        custom_keywords = [
            "science", "physics", "chemistry", "biology", "mathematics", "research", "scientific",
            "technology", "ai", "artificial intelligence", "machine learning", "blockchain", 
            "cryptocurrency", "programming", "computer", "software", "digital",
            "health", "medicine", "pandemic", "vaccine", "healthcare", "treatment", "medical",
            "disease", "virus", "bacteria", "infection", "drug", "therapy",
            "politics", "government", "election", "policy", "law", "political", "parliament", 
            "congress", "senate", "legislation", "propaganda"
        ]
        
        parser = WikiParser(
            dump_path=dump_path,
            output_dir="data/corpus",
            logger=logger,
            min_article_length=100,  # Skip very short articles
            max_article_length=50000,  # Skip extremely long articles
            batch_size=1000,  # Process 1000 articles at a time
            keywords=custom_keywords,  # Use custom keywords
            min_keyword_count=5  # Require more than 5 keyword instances in content
        )
        
        for stats in parser.parse_dump():
            if stats.get("completed"):
                logger.info(f"Processing completed: {stats}")
                final_stats = parser.get_corpus_stats()
                logger.info(f"Final statistics: {final_stats}")
            else:
                logger.info(f"Progress: {stats}")
    else:
        logger.error(f"Dump file not found: {dump_path}")