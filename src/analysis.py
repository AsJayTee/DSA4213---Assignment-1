"""
Vocabulary Statistics Visualizer
Analyzes vocab_stats.json from WikiParser and creates visualizations
Includes Mikolov et al. subsampling analysis
"""

import json
import math
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class VocabAnalyzer:
    """Analyze vocabulary statistics and create visualizations"""
    
    def __init__(self, vocab_stats_path: str):
        """
        Initialize with vocab_stats.json file
        
        Args:
            vocab_stats_path: Path to vocab_stats.json file
        """
        self.vocab_stats_path = Path(vocab_stats_path)
        self.stats = self._load_stats()
        self.word_frequencies = self.stats.get("word_frequencies", {})
        self.total_words = self.stats.get("total_words", 0)
        
        # Calculate relative frequencies
        self.word_freqs_rel = {
            word: count / self.total_words 
            for word, count in self.word_frequencies.items()
        }
    
    def _load_stats(self) -> dict:
        """Load vocabulary statistics from JSON file"""
        if not self.vocab_stats_path.exists():
            raise FileNotFoundError(f"Vocab stats file not found: {self.vocab_stats_path}")
        
        with open(self.vocab_stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def apply_mikolov_subsampling(self, threshold: float = 1e-5) -> dict[str, int]:
        """
        Apply Mikolov et al. subsampling to reduce frequent words
        
        Formula: P(w_i) = 1 - sqrt(t / f(w_i)) [discard probability]
        Keep probability = sqrt(t / f(w_i))
        where t is threshold and f(w_i) is word frequency
        
        Args:
            threshold: Subsampling threshold (default 1e-5)
            
        Returns:
            Dictionary of subsampled word frequencies
        """
        subsampled_vocab = {}
        
        for word, count in self.word_frequencies.items():
            freq = count / self.total_words
            
            if freq <= threshold:
                # Keep all instances of rare words
                subsampled_vocab[word] = count
            else:
                # Calculate keep probability (Mikolov formula)
                # P(discard) = 1 - sqrt(t / f), so P(keep) = sqrt(t / f)
                keep_prob = math.sqrt(threshold / freq)
                
                # Apply subsampling
                new_count = int(count * keep_prob)
                if new_count > 0:
                    subsampled_vocab[word] = new_count
        
        return subsampled_vocab
    
    def apply_alphabetic_filter(self) -> dict[str, int]:
        """
        Filter words to keep only those that are primarily alphabetic
        
        Removes:
        - Words containing symbols except apostrophes (') and hyphens (-)
        - Words that contain only numbers and symbols (no alphabetic characters)
        
        Returns:
            Dictionary of filtered word frequencies
        """
        import re
        
        alphabetic_vocab = {}
        
        for word, count in self.word_frequencies.items():
            # Check if word contains only allowed characters (letters, apostrophes, hyphens)
            if re.match(r"^[a-zA-Z'-]+$", word):
                # Additional check: must contain at least one alphabetic character
                if re.search(r"[a-zA-Z]", word):
                    alphabetic_vocab[word] = count
        
        return alphabetic_vocab
    
    def get_vocab_sizes_with_filters(self) -> dict[str, int]:
        """
        Calculate vocabulary sizes under different filtering conditions
        
        Returns:
            Dictionary mapping filter name to vocabulary size (in desired order)
        """
        from collections import OrderedDict
        
        sizes = OrderedDict()
        
        # Alphabetic filter
        alphabetic_filtered = self.apply_alphabetic_filter()
        
        # Minimum frequency filter
        min_freq_filtered = {w: c for w, c in self.word_frequencies.items() if c >= 20}
        
        # Combined filters
        combined_filtered = {w: c for w, c in alphabetic_filtered.items() if c >= 20}
        
        # Add in desired order
        sizes["Original"] = len(self.word_frequencies)
        sizes["Alphabetic Only"] = len(alphabetic_filtered)
        sizes["Min Freq ≥ 20"] = len(min_freq_filtered)
        sizes["Alphabetic + Min Freq ≥ 20"] = len(combined_filtered)
        
        return sizes
    
    def plot_vocabulary_sizes(self, save_path: str = None) -> None:
        """
        Create bar plot showing vocabulary sizes with different filters
        
        Args:
            save_path: Optional path to save the plot
        """
        sizes = self.get_vocab_sizes_with_filters()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Use square root scale for y-axis
        filter_names = list(sizes.keys())
        vocab_counts = list(sizes.values())
        sqrt_counts = [math.sqrt(count) for count in vocab_counts]
        
        # Create bars with a curated color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bar_colors = colors[:len(filter_names)]
        
        bars = plt.bar(filter_names, sqrt_counts, color=bar_colors, alpha=0.8, width=0.6)
        
        # Add horizontal threshold line at 1M vocabulary (sqrt(1M) = 1000)
        threshold_line = math.sqrt(1_000_000)
        plt.axhline(y=threshold_line, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='1M Vocabulary Threshold')
        
        # Customize the plot
        plt.xlabel('Vocabulary Filter', fontsize=12, fontweight='bold')
        plt.ylabel('√(Vocabulary Size)', fontsize=12, fontweight='bold')
        plt.title('Vocabulary Size Under Different Filtering Conditions', 
                 fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars (showing actual counts, not sqrt)
        for bar, count in zip(bars, vocab_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend for the threshold line
        plt.legend(loc='upper right')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Vocabulary sizes plot saved to: {save_path}")
        
        plt.show()
    
    def plot_word_frequency_distribution(self, save_path: str = None) -> None:
        """
        Plot word frequency distribution (Zipf's law)
        
        Args:
            save_path: Optional path to save the plot
        """
        # Get sorted frequencies
        sorted_freqs = sorted(self.word_frequencies.values(), reverse=True)
        ranks = range(1, len(sorted_freqs) + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Plot actual data on log-log scale to show Zipf's law
        plt.loglog(ranks, sorted_freqs, 'b-', alpha=0.7, linewidth=2, label='Actual Data')
        
        # Add reference line for perfect Zipf's law (slope = -1) across FULL range
        if len(ranks) > 1:
            # Calculate reference line across the entire range
            ref_y = [sorted_freqs[0] / r for r in ranks]
            plt.loglog(ranks, ref_y, 'r--', alpha=0.6, linewidth=2,
                      label='Perfect Zipf\'s Law (slope = -1)')
        
        plt.xlabel('Word Rank (log scale)', fontsize=12, fontweight='bold')
        plt.ylabel('Word Frequency (log scale)', fontsize=12, fontweight='bold')
        plt.title('Word Frequency Distribution (Zipf\'s Law Analysis)', 
                 fontsize=14, fontweight='bold')
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add some statistical info as text
        if len(sorted_freqs) > 2:
            # Calculate how well it follows Zipf's law (correlation in log space)
            log_ranks = np.log10(ranks)
            log_freqs = np.log10(sorted_freqs)
            correlation = np.corrcoef(log_ranks, log_freqs)[0, 1]
            
            plt.text(0.02, 0.98, f'Zipf Correlation: {correlation:.3f}\n(closer to -1.0 = better fit)', 
                    transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Frequency distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_vocabulary_distribution(self, save_path: str = None) -> None:
        """
        Plot the cumulative distribution of word usage to show vocabulary inequality
        Shows what percentage of total words are covered by top X% of vocabulary
        
        Args:
            save_path: Optional path to save the plot
        """
        # Get sorted frequencies (most common first)
        sorted_freqs = sorted(self.word_frequencies.values(), reverse=True)
        total_word_count = sum(sorted_freqs)
        vocab_size = len(sorted_freqs)
        
        # Calculate cumulative coverage
        cumulative_coverage = []
        cumulative_sum = 0
        
        for freq in sorted_freqs:
            cumulative_sum += freq
            coverage = (cumulative_sum / total_word_count) * 100
            cumulative_coverage.append(coverage)
        
        # Calculate percentage of vocabulary (x-axis)
        vocab_percentages = [(i + 1) / vocab_size * 100 for i in range(vocab_size)]
        
        plt.figure(figsize=(12, 8))
        
        # Plot the cumulative coverage curve
        plt.plot(vocab_percentages, cumulative_coverage, 'b-', linewidth=3, alpha=0.8)
        
        # Add reference lines to show inequality
        plt.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=2, 
                label='Perfect Equality Line')
        
        # Add some key statistics as vertical lines
        # Find what % of vocab covers 50%, 80%, 90% of words
        milestones = [50, 80, 90, 95]
        colors = ['green', 'orange', 'red', 'purple']
        
        for milestone, color in zip(milestones, colors):
            # Find the vocabulary percentage that covers this milestone
            idx = next((i for i, coverage in enumerate(cumulative_coverage) 
                       if coverage >= milestone), len(cumulative_coverage) - 1)
            vocab_pct = vocab_percentages[idx]
            
            plt.axvline(x=vocab_pct, color=color, linestyle=':', alpha=0.7, linewidth=2)
            plt.axhline(y=milestone, color=color, linestyle=':', alpha=0.7, linewidth=2)
            
            # Add annotation
            plt.annotate(f'{vocab_pct:.2f}% of vocabulary\ncovers {milestone}% of words',
                        xy=(vocab_pct, milestone), xytext=(vocab_pct + 10, milestone - 5),
                        fontsize=9, ha='left', va='top', color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Customize the plot
        plt.xlabel('Vocabulary Percentage (% of unique words)', fontsize=12, fontweight='bold')
        plt.ylabel('Coverage (% of total word occurrences)', fontsize=12, fontweight='bold')
        plt.title('Vocabulary Distribution: Cumulative Word Coverage\n(Shows how uneven word usage is)', 
                 fontsize=14, fontweight='bold')
        
        # Set limits and grid
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        
        # Add summary statistics in text box
        top_1_pct_idx = int(0.01 * vocab_size)
        top_10_pct_idx = int(0.10 * vocab_size)
        
        top_1_coverage = cumulative_coverage[top_1_pct_idx] if top_1_pct_idx < len(cumulative_coverage) else 0
        top_10_coverage = cumulative_coverage[top_10_pct_idx] if top_10_pct_idx < len(cumulative_coverage) else 0
        
        stats_text = f'Distribution Statistics:\n' \
                    f'• Top 1% of vocab: {top_1_coverage:.1f}% coverage\n' \
                    f'• Top 10% of vocab: {top_10_coverage:.1f}% coverage\n' \
                    f'• Total vocabulary: {vocab_size:,} words\n' \
                    f'• Total word count: {total_word_count:,}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Vocabulary distribution plot saved to: {save_path}")
        
        plt.show()
        """
        Visualize the effects of alphabetic and frequency filtering
        
        Args:
            save_path: Optional path to save the plot
        """
        # Apply different filters
        alphabetic_filtered = self.apply_alphabetic_filter()
        min_freq_filtered = {w: c for w, c in self.word_frequencies.items() if c >= 20}
        combined_filtered = {w: c for w, c in alphabetic_filtered.items() if c >= 20}
        
        # Get examples of filtered-out words
        removed_by_alphabetic = set(self.word_frequencies.keys()) - set(alphabetic_filtered.keys())
        removed_by_freq = set(self.word_frequencies.keys()) - set(min_freq_filtered.keys())
        
        # Sample some examples for display
        alphabetic_examples = sorted(
            [(word, self.word_frequencies[word]) for word in removed_by_alphabetic],
            key=lambda x: x[1], reverse=True
        )[:20]
        
        freq_examples = sorted(
            [(word, self.word_frequencies[word]) for word in removed_by_freq if word in alphabetic_filtered],
            key=lambda x: x[1], reverse=True
        )[:20]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Examples of words removed by alphabetic filter
        if alphabetic_examples:
            words1, counts1 = zip(*alphabetic_examples)
            y_pos1 = np.arange(len(words1))
            
            ax1.barh(y_pos1, counts1, color='lightcoral', alpha=0.8)
            ax1.set_yticks(y_pos1)
            ax1.set_yticklabels([f"'{word}'" for word in words1], fontsize=9)
            ax1.set_xlabel('Word Count', fontweight='bold')
            ax1.set_title('Top 20 Words Removed by\nAlphabetic Filter', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Examples of words removed by frequency filter (but kept by alphabetic)
        if freq_examples:
            words2, counts2 = zip(*freq_examples)
            y_pos2 = np.arange(len(words2))
            
            ax2.barh(y_pos2, counts2, color='lightskyblue', alpha=0.8)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(words2, fontsize=9)
            ax2.set_xlabel('Word Count', fontweight='bold')
            ax2.set_title('Top 20 Alphabetic Words Removed by\nMin Frequency ≥ 20 Filter', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Filtering effects plot saved to: {save_path}")
        
        plt.show()
    
    def print_summary_statistics(self) -> None:
        """Print comprehensive summary of vocabulary statistics"""
        print("=" * 60)
        print("VOCABULARY STATISTICS SUMMARY")
        print("=" * 60)
        
        # Basic stats
        print(f"Total words in corpus: {self.total_words:,}")
        print(f"Unique words (vocabulary size): {len(self.word_frequencies):,}")
        print(f"Articles processed: {self.stats.get('articles_processed', 'N/A'):,}")
        
        if self.stats.get('articles_keyword_filtered'):
            print(f"Articles filtered by keywords: {self.stats['articles_keyword_filtered']:,}")
            print(f"Filter rate: {self.stats.get('filter_rate', 'N/A')}")
        
        print()
        
        # Word frequency stats
        freqs = list(self.word_frequencies.values())
        print("WORD FREQUENCY STATISTICS:")
        print(f"Mean frequency: {np.mean(freqs):.2f}")
        print(f"Median frequency: {np.median(freqs):.2f}")
        print(f"Max frequency: {max(freqs):,}")
        print(f"Min frequency: {min(freqs):,}")
        print()
        
        # Top words
        top_words = sorted(self.word_frequencies.items(), 
                          key=lambda x: x[1], reverse=True)[:10]
        print("TOP 10 MOST FREQUENT WORDS:")
        for i, (word, count) in enumerate(top_words, 1):
            freq_pct = (count / self.total_words) * 100
            print(f"{i:2d}. {word:<15} {count:>8,} ({freq_pct:.3f}%)")
        print()
        
        # Vocabulary size analysis
        sizes = self.get_vocab_sizes_with_filters()
        print("VOCABULARY SIZES WITH DIFFERENT FILTERS:")
        for filter_name, size in sizes.items():
            reduction = ((len(self.word_frequencies) - size) / len(self.word_frequencies)) * 100
            print(f"{filter_name:<25} {size:>8,} ({reduction:5.1f}% reduction)")
        print()
        
        # Alphabetic filter analysis
        alphabetic_filtered = self.apply_alphabetic_filter()
        non_alphabetic_count = len(self.word_frequencies) - len(alphabetic_filtered)
        non_alphabetic_pct = (non_alphabetic_count / len(self.word_frequencies)) * 100
        
        print("ALPHABETIC FILTER ANALYSIS:")
        print(f"Non-alphabetic words removed: {non_alphabetic_count:,} ({non_alphabetic_pct:.1f}%)")
        
        # Show examples of removed words
        removed_words = set(self.word_frequencies.keys()) - set(alphabetic_filtered.keys())
        if removed_words:
            # Get some examples of removed words, sorted by frequency
            removed_examples = sorted(
                [(word, self.word_frequencies[word]) for word in removed_words],
                key=lambda x: x[1], reverse=True
            )[:10]
            print("Examples of removed non-alphabetic words:")
            for word, count in removed_examples:
                print(f"  '{word}' ({count:,} occurrences)")
        print()
        
        # Combined filter effect
        combined_filtered = {w: c for w, c in alphabetic_filtered.items() if c >= 20}
        combined_reduction = ((len(self.word_frequencies) - len(combined_filtered)) / len(self.word_frequencies)) * 100
        print("COMBINED FILTER EFFECT (Alphabetic + Min Freq ≥ 20):")
        print(f"Total vocabulary reduction: {len(self.word_frequencies) - len(combined_filtered):,} ({combined_reduction:.1f}%)")
        print(f"Final vocabulary size: {len(combined_filtered):,}")
        
        # Mikolov subsampling analysis (moved to end, less emphasis)
        subsampled = self.apply_mikolov_subsampling()
        total_subsampled_words = sum(subsampled.values())
        word_reduction = ((self.total_words - total_subsampled_words) / self.total_words) * 100
        
        print("\nMIKOLOV SUBSAMPLING (affects word counts, not vocab size):")
        print(f"Total word count reduction: {self.total_words - total_subsampled_words:,} ({word_reduction:.1f}%)")
        
    def plot_vocabulary_distribution(self, save_path: str = None) -> None:
        """
        Plot the cumulative distribution of word usage to show vocabulary inequality
        Shows what percentage of total words are covered by top X% of vocabulary
        
        Args:
            save_path: Optional path to save the plot
        """
        # Get sorted frequencies (most common first)
        sorted_freqs = sorted(self.word_frequencies.values(), reverse=True)
        total_word_count = sum(sorted_freqs)
        vocab_size = len(sorted_freqs)
        
        # Calculate cumulative coverage
        cumulative_coverage = []
        cumulative_sum = 0
        
        for freq in sorted_freqs:
            cumulative_sum += freq
            coverage = (cumulative_sum / total_word_count) * 100
            cumulative_coverage.append(coverage)
        
        # Calculate percentage of vocabulary (x-axis)
        vocab_percentages = [(i + 1) / vocab_size * 100 for i in range(vocab_size)]
        
        plt.figure(figsize=(12, 8))
        
        # Plot the cumulative coverage curve
        plt.plot(vocab_percentages, cumulative_coverage, 'b-', linewidth=3, alpha=0.8)
        
        # Add reference lines to show inequality
        plt.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=2, 
                label='Perfect Equality Line')
        
        # Add some key statistics as vertical lines
        # Find what % of vocab covers 50%, 80%, 90% of words
        milestones = [50, 80, 90, 95]
        colors = ['green', 'orange', 'red', 'purple']
        
        for milestone, color in zip(milestones, colors):
            # Find the vocabulary percentage that covers this milestone
            idx = next((i for i, coverage in enumerate(cumulative_coverage) 
                       if coverage >= milestone), len(cumulative_coverage) - 1)
            vocab_pct = vocab_percentages[idx]
            
            plt.axvline(x=vocab_pct, color=color, linestyle=':', alpha=0.7, linewidth=2)
            plt.axhline(y=milestone, color=color, linestyle=':', alpha=0.7, linewidth=2)
            
            # Add annotation
            plt.annotate(f'{vocab_pct:.2f}% of vocabulary\ncovers {milestone}% of words',
                        xy=(vocab_pct, milestone), xytext=(vocab_pct + 10, milestone - 5),
                        fontsize=9, ha='left', va='top', color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Customize the plot
        plt.xlabel('Vocabulary Percentage (% of unique words)', fontsize=12, fontweight='bold')
        plt.ylabel('Coverage (% of total word occurrences)', fontsize=12, fontweight='bold')
        plt.title('Vocabulary Distribution: Cumulative Word Coverage\n(Shows how uneven word usage is)', 
                 fontsize=14, fontweight='bold')
        
        # Set limits and grid
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        
        # Add summary statistics in text box
        top_1_pct_idx = int(0.01 * vocab_size)
        top_10_pct_idx = int(0.10 * vocab_size)
        
        top_1_coverage = cumulative_coverage[top_1_pct_idx] if top_1_pct_idx < len(cumulative_coverage) else 0
        top_10_coverage = cumulative_coverage[top_10_pct_idx] if top_10_pct_idx < len(cumulative_coverage) else 0
        
        stats_text = f'Distribution Statistics:\n' \
                    f'• Top 1% of vocab: {top_1_coverage:.1f}% coverage\n' \
                    f'• Top 10% of vocab: {top_10_coverage:.1f}% coverage\n' \
                    f'• Total vocabulary: {vocab_size:,} words\n' \
                    f'• Total word count: {total_word_count:,}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Vocabulary distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_filtering_effects(self, save_path: str = None) -> None:
        """
        Visualize the effects of alphabetic and frequency filtering
        
        Args:
            save_path: Optional path to save the plot
        """
        # Apply different filters
        alphabetic_filtered = self.apply_alphabetic_filter()
        min_freq_filtered = {w: c for w, c in self.word_frequencies.items() if c >= 20}
        combined_filtered = {w: c for w, c in alphabetic_filtered.items() if c >= 20}
        
        # Get examples of filtered-out words
        removed_by_alphabetic = set(self.word_frequencies.keys()) - set(alphabetic_filtered.keys())
        removed_by_freq = set(self.word_frequencies.keys()) - set(min_freq_filtered.keys())
        
        # Sample some examples for display
        alphabetic_examples = sorted(
            [(word, self.word_frequencies[word]) for word in removed_by_alphabetic],
            key=lambda x: x[1], reverse=True
        )[:20]
        
        freq_examples = sorted(
            [(word, self.word_frequencies[word]) for word in removed_by_freq if word in alphabetic_filtered],
            key=lambda x: x[1], reverse=True
        )[:20]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Examples of words removed by alphabetic filter
        if alphabetic_examples:
            words1, counts1 = zip(*alphabetic_examples)
            y_pos1 = np.arange(len(words1))
            
            ax1.barh(y_pos1, counts1, color='lightcoral', alpha=0.8)
            ax1.set_yticks(y_pos1)
            ax1.set_yticklabels([f"'{word}'" for word in words1], fontsize=9)
            ax1.set_xlabel('Word Count', fontweight='bold')
            ax1.set_title('Top 20 Words Removed by\nAlphabetic Filter', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Examples of words removed by frequency filter (but kept by alphabetic)
        if freq_examples:
            words2, counts2 = zip(*freq_examples)
            y_pos2 = np.arange(len(words2))
            
            ax2.barh(y_pos2, counts2, color='lightskyblue', alpha=0.8)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(words2, fontsize=9)
            ax2.set_xlabel('Word Count', fontweight='bold')
            ax2.set_title('Top 20 Alphabetic Words Removed by\nMin Frequency ≥ 20 Filter', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Filtering effects plot saved to: {save_path}")
        
        plt.show()
    
    def create_all_visualizations(self, output_dir: str = "visualizations") -> None:
        """
        Create all visualizations and save to specified directory
        
        Args:
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Creating vocabulary visualizations...")
        
        # Print summary first
        self.print_summary_statistics()
        
        # Create all plots
        self.plot_vocabulary_sizes(output_path / "vocab_sizes_comparison.png")
        self.plot_word_frequency_distribution(output_path / "word_frequency_distribution.png")
        self.plot_vocabulary_distribution(output_path / "vocabulary_distribution.png")
        self.plot_filtering_effects(output_path / "filtering_effects.png")
        
        print(f"\nAll visualizations saved to: {output_path}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Analyze vocabulary statistics from WikiParser")
    parser.add_argument("vocab_file", help="Path to vocab_stats.json file")
    parser.add_argument("--output", "-o", default="visualizations", 
                       help="Output directory for plots (default: visualizations)")
    parser.add_argument("--threshold", "-t", type=float, default=1e-5,
                       help="Mikolov subsampling threshold (default: 1e-5)")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = VocabAnalyzer(args.vocab_file)
        
        # Create all visualizations
        analyzer.create_all_visualizations(args.output)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid path to vocab_stats.json file")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Example usage for direct execution
    # Replace with actual path to your vocab_stats.json file
    vocab_file = "data/corpus/WikiDump-2019-vocab_stats.json"
    
    if Path(vocab_file).exists():
        analyzer = VocabAnalyzer(vocab_file)
        analyzer.create_all_visualizations()
    else:
        print(f"Example vocab file not found: {vocab_file}")
        print("Usage: python vocab_visualizer.py path/to/vocab_stats.json")
        print("Or use command line: python vocab_visualizer.py --help")