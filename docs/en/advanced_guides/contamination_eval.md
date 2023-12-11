# Contamination Evaluation Guidance

**Data contamination**, i.e.,
the presence of test data from these downstream tasks in the pre-training data of LLMs, may inflate LLM performance observed on many downstream tasks (e.g., summarization, natural language inference, text classification).

To evaluate LLM with contaminated data, we employed [Contamination Detector](https://github.com/liyucheng09/Contamination_Detector) to generate contamination labels.

## Introduction to [Detection Tools](https://github.com/liyucheng09/Contamination_Detector)

Contamination Detector aids in identifying and analyzing such potential contamination without requiring access to the LLMs' training data based on Internet presence verification, enabling even small teams and individuals to conduct robust evaluation.

### Method

- Using the Bing Search API to check if verbatim test examples appear online, which likely indicates inclusion in Common Crawl.

- Specifically verifying if pages containing verbatim test examples were indexed in the 2017-2020 Common Crawl, by only searching the URLs rather than
  full contents.

#### Construct queries

for example:
**Question**: The flaw in Anderson’s ACT
theory was that some considered it \_\_\_\_.
**Choices**:
A: ’Only applicable to a motor system’,
B: ’Untestable and thus, of uncertain sci-
entific value’,
C: ’Lacking in definition for its ele-
ments’
D: ’Overly complex in explaining the
operation of cognition’,
**Answer**: B
**Query**: The flaw in Anderson’s ACT theory was that some considered it untestable and thus, of uncertain scientific value.

#### Improve Matching

To avoid potential false positives, the method is configured with two key settings:

- an order penalty (gamma of 0.8) for METEOR ensures matches respect sequence;
- matching is constrained to a window up
  to 2x the query length, preventing partial or out-of-
  context matches.

#### Contamination Type

- *input contamination* where only question is presented in the
  matched pages but not answer;
- *input-and-label contamination* where both question and answer occur in the matched pages.

## Data Preparation

To be complete

## Evaluation Configuration

To be complete
