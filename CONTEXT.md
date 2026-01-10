# Technical Design Document: LexiconWeaver
**Version:** 1.0
**Status:** Approved for Development
**Scope:** A Robust, Human-in-the-Loop Framework for Web Novel Translation

---

## 1. Executive Summary & Core Philosophy

**LexiconWeaver** is a specialized translation framework designed to address the specific shortcomings of Machine Translation (MT) in the context of Web Novels (Xianxia, LitRPG, Fantasy).

### The Problem
Generic translators optimize for sentence-level fluency, ignoring novel-wide consistency. This results in "Term Drift"—where a proper noun like "Spirit Severing" is translated as "Spirit Cutting" in Chapter 1 and "Soul Split" in Chapter 2.

### The Solution
LexiconWeaver introduces a **Middleware Layer** that enforces terminology constraints *before* the AI generates text. It relies on a "Human-in-the-Loop" workflow:
1.  **Scout:** The machine identifies potential terms.
2.  **Annotate:** The human defines the terms (once).
3.  **Weave:** The machine generates text strictly adhering to those definitions.

---

## 2. System Architecture

The application is built as a modular Monolith using Python. It consists of three primary engines that interact via a central SQLite database.

### 2.1 The Three Engines
1.  **The Scout (Discovery Engine):** A heuristic-based analyzer that ingests raw text and outputs "Candidate Terms" with confidence scores. It does not translate; it identifies *what needs translating*.
2.  **The Annotator (Interaction Engine):** A Terminal User Interface (TUI) built with **Textual**. It manages the user's workflow, allowing them to confirm, reject, or edit candidates.
3.  **The Weaver (Generation Engine):** The pipeline that interfaces with the Local LLM (Ollama). It manages context injection, prompt construction, and post-translation verification.

### 2.2 Data Flow
`Raw Text File` -> **[Scout]** -> `Candidate List` -> **[Annotator (User)]** -> `Glossary DB` -> **[Weaver]** -> `Translated Text`

---

## 3. Detailed Component Logic

### 3.1 The Scout: Heuristic Discovery
The Scout avoids "AI Hallucinations" by relying on deterministic algorithms first, and lightweight AI second. It calculates a **Confidence Score (0.0 - 1.0)** for every potential term.

**A. The Scoring Algorithm**
The Scout scans the text and assigns points based on these factors:
1.  **Frequency Weight (30%):** How often does a phrase appear?
    * *Logic:* A phrase appearing 50 times is more likely to be a term than one appearing twice.
2.  **Capitalization Weight (30%):** Does the phrase appear Capitalized in the middle of a sentence?
    * *Logic:* "The Golden Core" vs "the golden core". Capitalization is a strong signal for Proper Nouns.
3.  **Structural Context Weight (40%):** Does the phrase appear in a "Definition Pattern"?
    * *Pattern 1:* "called [X]" (e.g., "a technique called **Void Step**")
    * *Pattern 2:* "rank [X]" (e.g., "**Rank 3** Gu")
    * *Pattern 3:* "known as [X]"
    * *Logic:* Any word appearing in these slots is almost certainly a special term, even if it appears only once.

**B. The Filtration System**
Before showing results to the user, the Scout filters noise:
* **Stopword Filter:** Checks against a local hash set of 50,000 common English words.
* **Ignore List:** Checks against the user's project-specific "Ignored Terms" database (words the user previously marked as "Not a term").

---

### 3.2 The Annotator: TUI & Term Management
This is the user's workspace. It focuses on speed and keyboard efficiency.

**A. Hierarchical Term Resolution**
The system must handle overlapping terms.
* *Scenario:* You have defined "Gu" and "Gu Master".
* *Logic:* The system enforces a **Longest Match First** rule. When processing text, it prioritizes "Gu Master" over "Gu". This prevents the system from translating "Gu Master" as "[Insect] Master".

**B. Term Metadata**
When a user approves a term, the following data is captured:
1.  **Source:** The original string (e.g., "Heavenly Court").
2.  **Target:** The translation (e.g., "Cennet Mahkemesi").
3.  **Category:** (Optional) "Person", "Location", "Skill", "Item". This helps the LLM understand context.
4.  **Scope:**
    * *Global (Default):* Applies to the entire novel.
    * *Local:* Applies only to specific chapters (user-defined range).

**C. The Interface (Textual)**
* **Left Panel:** Displays the Chapter Text.
    * *Green Highlight:* Terms already in the DB.
    * *Yellow Highlight:* High-confidence candidates found by the Scout.
* **Right Panel:** The "Candidate Queue".
    * Sorted by Confidence Score.
    * Keybindings: `Enter` to Edit/Confirm, `Del` to Ignore, `S` to Skip.

---

### 3.3 The Weaver: Context-Aware Translation
The Weaver is responsible for the final output. It treats the LLM as an "unreliable worker" that needs strict instructions.

**A. Dynamic Glossary Injection**
We cannot feed the entire 5,000-word glossary to the LLM for every paragraph (it wastes context window tokens).
* *Step 1:* Isolate the current paragraph.
* *Step 2:* Scan the paragraph using **FlashText** (Aho-Corasick algorithm) to find which glossary terms strictly appear in this specific text.
* *Step 3:* Construct a "Mini-Glossary" containing *only* those terms.
* *Step 4:* Append this Mini-Glossary to the System Prompt.

**B. The Prompt Strategy**
The prompt sent to Ollama follows this structure:
1.  **Role:** "You are a fantasy translator."
2.  **Constraint:** "You must use the provided glossary mappings exactly."
3.  **Glossary Block:** (The Mini-Glossary found in Step A).
4.  **Input:** The raw paragraph.

**C. Post-Processing Verification (The Safety Net)**
After the LLM returns the translated text, the Weaver runs a sanity check.
* *Logic:* "Did the input paragraph contain 'All Out Gu'? Yes. Did the output text contain 'Bütün Güç Gu'? No."
* *Action:* If a check fails, the paragraph is flagged as **[UNSTABLE]** in the UI. The user must review it manually. The system never "auto-corrects" because it might break grammar.

---

## 4. Data Design (SQLite Schema)

The database is designed for referential integrity. We use SQLite because it is serverless, fast, and easy to backup.

### Table: `projects`
* `id` (PK): Unique Identifier.
* `title`: Name of the novel.
* `source_lang`: Source language code (e.g., "en").
* `target_lang`: Target language code (e.g., "tr").
* `created_at`: Timestamp.

### Table: `glossary_terms`
* `id` (PK): Unique Identifier.
* `project_id` (FK): Links to `projects`.
* `source_term`: The English term (Indexed for speed).
* `target_term`: The translated term.
* `category`: Text field (Skill, Item, Name).
* `is_regex`: Boolean. If true, `source_term` is treated as a Regex pattern (e.g., "Rank \d").
* `confidence`: Float (Stored for historical analytics).

### Table: `ignored_terms`
* `id` (PK): Unique Identifier.
* `project_id` (FK): Links to `projects`.
* `term`: The word to be permanently ignored by the Scout.

### Table: `translation_cache`
* `hash` (PK): SHA-256 hash of the *source paragraph*.
* `project_id` (FK): Links to `projects`.
* `translation`: The accepted translation string.
* *Purpose:* If you re-run the program on Chapter 1, it fetches these results instantly instead of burning GPU power re-translating.

---

## 5. Implementation Roadmap

### Phase 1: The Core Logic (No UI)
* **Goal:** A Python script that takes a text file and a hardcoded dictionary, translates it, and verifies terms.
* **Deliverables:**
    * `DatabaseManager` class (SQLite setup).
    * `Scout` class (Frequency & N-gram analysis).
    * `Weaver` class (Ollama connection & Prompt builder).

### Phase 2: The TUI Skeleton
* **Goal:** A working Textual application that can display text.
* **Deliverables:**
    * Main Screen Layout (Side-by-side panels).
    * Text loading & scrolling.
    * Basic "Find & Highlight" functionality (coloring words based on a list).

### Phase 3: Interaction & Binding
* **Goal:** Connecting the UI to the Logic.
* **Deliverables:**
    * "Add Term" Modal Dialog.
    * Connecting the "Scout" results to the Right Panel list.
    * Saving user choices to the SQLite DB.

### Phase 4: Production Polish
* **Goal:** Stability and User Experience.
* **Deliverables:**
    * Streaming response support (seeing the translation type out in real-time).
    * Progress bars for long chapters.
    * Error handling (What if Ollama is offline?).

---

## 6. Technology Stack & Requirements

### Core Technologies
* **Python 3.12:** The engine.
* **Textual:** The TUI framework.
* **Peewee ORM:** For simple, Pythonic database interaction.
* **Ollama:** The local LLM server.

### Key Libraries
* `spacy`: Used for "Part of Speech" tagging (distinguishing Nouns from Verbs).
* `flashtext`: Used for the O(N) term replacement algorithm (The "Aho-Corasick" implementation).
* `rich`: Used for beautiful text formatting (colors, bolding) inside the terminal.
* `httpx`: For fast, async communication with the Ollama API.

---

## 7. Resilience & Error Handling

To make this "Production Level," we must anticipate failure.

* **LLM Timeout:** If Ollama hangs, the Weaver handles the timeout gracefully, retries once, and then pauses the queue, alerting the user.
* **Database Lock:** SQLite can lock if accessed by multiple threads. We will use a "Singleton" database connection pattern to ensure only one write operation happens at a time.
* **Crash Recovery:** Since we save every defined term to the DB immediately, and we cache every translated paragraph, a crash means zero data loss. The user simply restarts the app and resumes from the exact sentence they were on.