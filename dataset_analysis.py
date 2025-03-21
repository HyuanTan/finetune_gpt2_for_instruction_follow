import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from scripts.tools import load_file
import os

def main():
    plots_path = "./plots/"
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    dataset = "alpaca_data_52k.json"
    # dataset = "instruction_data_1k.json"
    file_path = f"./dataset/{dataset}"
    data = load_file(file_path)
    data = data[:2000]
    print("Number of entries:", len(data))
    
    instruction_lengths = []
    input_lengths = []
    output_lengths = []
    input_empty_count = 0
    instruction_texts = []

    # Step 2: Process each sample
    for sample in data:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')

        instruction_lengths.append(len(instruction.split()))
        input_lengths.append(len(input_text.split()))
        output_lengths.append(len(output_text.split()))
        instruction_texts.append(instruction)

        if input_text.strip() == "":
            input_empty_count += 1

    # Step 3: Estimate instruction category using keyword-based classification
    instruction_categories = []
    category_keywords = {
        "translation": ["translate", "translation"],
        "summarization": ["summarize", "summary"],
        "classification": ["classify", "classification"],
        "qa": ["question", "answer"],
        "reasoning": ["why", "how"],
        "generation": ["write", "generate"],
        "math": ["calculate", "math", "compute", "convert"],
        "correction": ["edit", "grammar", "spelling", "correct"],
        "coding": ["code", "program"],
        "general": []
    }

    for instr in instruction_texts:
        found = False
        for category, keywords in category_keywords.items():
            if any(re.search(rf"\b{kw}\b", instr.lower()) for kw in keywords):
                instruction_categories.append(category)
                found = True
                break
        if not found:
            instruction_categories.append("general")

    # Step 4: Calculate statistics
    category_counts = Counter(instruction_categories)
    input_empty_ratio = input_empty_count / len(data)

    # Summary statistics table
    summary = {
        "Total samples": len(data),
        "Empty input samples": input_empty_count,
        "Empty input ratio": round(input_empty_ratio, 4),
        "Average instruction length": round(sum(instruction_lengths) / len(instruction_lengths), 2),
        "Average input length": round(sum(input_lengths) / len(input_lengths), 2),
        "Average output length": round(sum(output_lengths) / len(output_lengths), 2),
        "Instruction category counts": dict(category_counts)
    }
    summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(instruction_texts)
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
    word_freq_sorted = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    word_freq_df = pd.DataFrame(list(word_freq_sorted.items()), columns=['Word/N-gram', 'Frequency'])


    # Step 5: Visualize instruction length distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(instruction_lengths, bins=10, kde=True)
    plt.title("Instruction Length Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{plots_path}instruction_length_distribution.png")

    # Step 6: Visualize input empty ratio
    plt.figure(figsize=(5, 5))
    plt.pie([input_empty_ratio, 1 - input_empty_ratio],
            labels=["Empty Input", "Non-empty Input"],
            autopct='%1.1f%%', startangle=140)
    plt.title("Empty Input Proportion")
    plt.tight_layout()
    plt.savefig(f"{plots_path}input_empty_ratio.png")

    # Step 7: Visualize instruction category distribution
    plt.figure(figsize=(8, 5))
    category_df = pd.DataFrame.from_dict(category_counts, orient='index', columns=['Count']).sort_values(by='Count', ascending=False)
    sns.barplot(x=category_df.index, y=category_df['Count'])
    plt.title("Instruction Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plots_path}instruction_category_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title("Instruction Dataset Summary")
    plt.tight_layout()
    plt.savefig(f"{plots_path}instruction_summary_table.png")
     

    # Step 8: Word Frequency and N-gram analysis (unigrams and bigrams)
    plt.figure(figsize=(10, 5))
    top_words = word_freq_df.head(10)
    sns.barplot(x='Word/N-gram', y='Frequency', data=top_words)
    plt.xticks(rotation=45)
    plt.title("Top 10 Word/N-gram Frequencies in Instructions")
    plt.tight_layout()
    plt.savefig(f"{plots_path}word_ngram_top10.png")
 
    
    # Step 9: Instruction Embedding Visualization using PCA (faster and better for small samples)
    tfidf = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf.fit_transform(instruction_texts).toarray()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf)

    # Plot PCA result
    plt.figure(figsize=(8, 6))
    for i, category in enumerate(set(instruction_categories)):
        idx = [j for j, x in enumerate(instruction_categories) if x == category]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=category)
    plt.legend()
    plt.title("Instruction Embedding Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(f"{plots_path}instruction_pca_visualization.png")

if __name__ == "__main__":
    main()