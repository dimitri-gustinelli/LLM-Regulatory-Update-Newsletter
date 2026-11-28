# AI-Powered IRRBB Regulatory Monitoring

The notebook shows how AI can support regulatory monitoring and impact assessment in banking risk management, particularly for treasury, ALM, IRRBB and regulatory compliance teams. The workflow compares individual provisions against an existing regulatory knowledge base (KB) to distinguish substantive amendments from renumbering and formatting changes. It combines web scraping, semantic retrieval and large language models (LLMs) to identify changes in banking regulation and generate practitioner-oriented newsletters with article-level references.


**Current scope:** Basel Committee on Banking Supervision (BCBS). Extensions to EBA, EU and US regulatory frameworks is planned.

## Workflow

<p align="center">
  <img src="./BCBS%20IRRBB%20Regulatory%20Workflow.png" width="900">
  <br>
</p>


### 1. Regulatory publication monitoring

The notebook scans the [BIS publications website](https://www.bis.org/publications) within a configurable date window and identifies relevant publications using case-insensitive keyword matching on publication titles.

### 2. Redline-aware PDF extraction

Standard PDF extraction loses visual information that is essential to understanding regulatory amendments. A custom `RedlinePDFLoader`, built with PyMuPDF and LangChain, preserves this information by detecting red text and strikethrough annotations or vector lines:

| PDF formatting | Extracted representation | Interpretation |
| --- | --- | --- |
| Red text | `<ins>...</ins>` | Newly inserted wording |
| Strikethrough text | `<del>...</del>` | Deleted wording |
| Ordinary text | Untagged text | Baseline wording |

These tags are passed to the LLM as part of the regulatory text, with explicit instructions on how to interpret them.

### 3. Article-level chunking

The custom `chunk_by_article()` function uses regulatory article identifiers (e.g. `31.36`, `98.49`, `Article 5`) instead of fixed token lengths to segment documents.

The reference knowledge base is a snapshot of the [Basel Framework](https://www.bis.org/committees/bcbs/basel-framework), restricted to:

- **SRP31:** Interest rate risk in the banking book.
- **SRP98:** Application guidance for IRRBB.
- **DIS70:** IRRBB disclosure requirements.

### 4. Semantic retrieval and comparison

The `thenlper/gte-large` sentence-transformer model converts each provision into a 1,024-dimensional embedding. The KB embeddings are stored locally in a Chroma vector database.

For each new chunk, the workflow retrieves its nearest KB counterpart (`K = 1`). Semantic retrieval helps align provisions even when their article numbers have changed. The resulting pairs are filtered using an experimental Chroma distance threshold of `0.016503` (lower scores indicate greater similarity).

Distance is used as a **screening heuristic**, not as evidence that a regulatory obligation has or has not changed.

### 5. Two-agent LLM pipeline

| Agent | Input | Responsibility | Output |
| --- | --- | --- | --- |
| **Regulatory Analyst** | `new_text`, `kb_text` | Compare obligations, scope, definitions, thresholds, formulas and reporting requirements; distinguish amendments from formatting or renumbering | Structured JSON for each article pair |
| **Newsletter Editor** | Flagged changes, article references and a human-provided template | Consolidate related amendments into thematic, practitioner-focused takeaways | Regulatory newsletter with article references |

The Regulatory Analyst agent is explicitly instructed to respect insertion/deletion tags, avoid unsupported claims and flag ambiguous cases for human review.

Example analyst output schema:

```json
{
  "new_article_ref": "31.44",
  "kb_article_ref": "31.43",
  "has_change": true,
  "change_summary": "Concise description of the substantive amendment."
}
```

The Newsletter Editor analyses only rows flagged as changes, together with the article references. A human-provided template guides the tone, structure and framing of the final newsletter. The pipeline uses an OpenAI-compatible API; the demonstration is configured with `gpt-6-luna` and temperature `1`.

## Case study: BCBS additional Pillar 2 guidance on IRRBB

The notebook demonstrates the workflow using the consultation on [Interest rate risk in the banking book – additional Pillar 2 guidance](https://www.bis.org/publications/interest-rate-risk-banking-book---additional-pillar-2-guidance), identified in the publication search dated 11 November 2026.

The generated newsletter highlights themes including:

- Non-maturity deposit (NMD) segmentation and evidence supporting behavioural assumptions.
- Backtesting, independent model validation and recalibration.
- Earnings-based risk indicators, stress testing and embedded gains or losses on securities.
- Supervisory assessment of capital and liquidity positions, and peer-group comparisons.

### Experimental results

| Screening stage | Article/chunk pairs |
| --- | ---: |
| Initial nearest-neighbour matches | 214 |
| Excluded due to conflicting leading numbers | 4 |
| Pairs remaining after chapter filtering | 210 |
| Selected for LLM analysis (distance > 0.016503) | 82 |
| Below or equal to the threshold | 128 |

The distance filter reduced the primary LLM comparison set by approximately **61%** relative to the 210 remaining pairs. To test its reliability, the notebook also ran the same LLM pipeline on the 128 excluded pairs.

Two excluded pairs were flagged as changes. The notebook's subsequent inspection identifies one genuine amendment (SRP31.36, sensitivity testing) and one false positive (SRP31.129, term-deposit redemption ratio) caused by imperfect equation extraction. This illustrates the trade-off between screening efficiency and missed amendments; it is not a comprehensive accuracy benchmark or a measured API cost saving.

## Getting started

### Prerequisites

- Jupyter Notebook or JupyterLab.
- Internet access for BIS publications, the embedding model and LLM API calls.
- An OpenAI API key and endpoint with access to a compatible chat-completions model.
- A pre-amendment Basel Framework PDF containing SRP31, SRP98 and DIS70, saved in the notebook's working directory as `basel-framework-IRRBB-SRP31-SRP98-DIS70.pdf`.

### API configuration

Create a local `config.json` file in the notebook's working directory:

```json
{
  "OPENAI_API_KEY": "your-api-key",
  "OPENAI_API_BASE": "https://api.openai.com/v1"
}
```
API keys (secret keys) are obtained here: https://platform.openai.com/api-keys

Choose the `MODEL_NAME` among models available on https://developers.openai.com/api/docs/models

**Do not commit `config.json` or API credentials to GitHub.** 

### Execution

Ensure that `config.json` and `basel-framework-IRRBB-SRP31-SRP98-DIS70.pdf` are located in the same working directory of the notebook.

1. Open `Reg_Update_IRRBB.ipynb` in Jupyter.
2. Set the monitoring parameters:

   ```python
   REFERENCE_DATE = "11.11.26"  # dd.mm.yy; historical case-study date
   LOOKBACK_DAYS = 7
   KEYWORDS = ["IRRBB", "interest rate risk"]
   ```

3. Run the publication discovery and PDF download cells. The current implementation downloads the first matching publication. If no publication is found, stop the notebook manually.
4. Execute the redline extraction, article chunking and KB embedding cells. Use a fresh `IRRBB_REG_db` directory when changing the KB snapshot or embedding configuration.
5. Run the matching and filtering cells, review questionable matches and update the publication-specific newsletter template in `generate_newsletter()`.
6. Execute the LLM pipeline and inspect its output against the original regulatory text. The final notebook cells optionally repeat the analysis on the low-distance excluded set.

```python
analyzed_df, final_newsletter = run_regulatory_pipeline(df=matches_filt_df)
print(final_newsletter)
```

## Outputs

| Output | Description |
| --- | --- |
| `IRRBB_REG_db/` | Local persistent Chroma knowledge base |
| `analyzed_df` | Compared text pairs, new and prior article references, change flags and summary of regulatory change |
| `final_newsletter` | Generated newsletter as a text string displayed in the notebook |
| `regulatory_analysis_results.xlsx` | Export of the primary article-level analysis |
| `regulatory_analysis_discarded_results.xlsx` | Diagnostic analysis of pairs excluded by the distance threshold |

## Limitations and roadmap

This is an exploratory proof of concept rather than a production-ready regulatory monitoring system.

| Area | Limitation / planned improvement |
| --- | --- |
| PDF extraction | Improve equation and table handling, recognition of red-formatted article headings, and removal of headers, footers and page numbers. Equation extraction is a high-priority issue. |
| Retrieval | Refine the distance metric and matching logic to better distinguish legally significant wording changes (e.g. *must* vs. *should*) from semantically similar text, including new or relocated provisions. |
| Validation | Introduce an independent validation agent, ideally using a different model, to review excluded pairs and return missed amendments to the editor. |
| Prompt engineering | Reduce dependence on the human template, separate verified publication metadata from stylistic guidance and evaluate models with lower-temperature settings. |
| Knowledge base | Add regulator, publication date, effective date and validity-period metadata; assess whether redline publications can be analysed directly without a historical KB. |
| Automation | Implement scheduled execution, rolling date windows, robust no-update stop conditions and broader publication filtering. |
| Coverage | Extend beyond BCBS to EBA Guidelines, RTS/ITS, EUR-Lex and US regulation. |

## Technology stack

**Python · Jupyter · Requests · Beautiful Soup · PyMuPDF · LangChain · Sentence Transformers · GTE-large · ChromaDB · OpenAI-compatible API · pandas**

## Disclaimer

This project is intended for research and educational purposes. LLM-generated analyses and newsletters may contain omissions, inaccurate interpretations or unsupported claims. They should be checked against the original regulatory publications and should not be relied upon as legal, regulatory or supervisory advice. The workflow does not replace expert review.
