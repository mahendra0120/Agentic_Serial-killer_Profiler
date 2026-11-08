# Agentic_Serial-killer_Profiler

*Project Report · Prepared 6 November 2026 · Sources: GitHub [`mahendra0120/Agentic_Serial-killer_Profiler`](https://github.com/mahendra0120/Agentic_Serial-killer_Profiler) (code) and Hugging Face [`mahendra0120/Agentic_Serial-killer_Profiler`](https://huggingface.co/mahendra0120/Agentic_Serial-killer_Profiler) (code plus model cache)*

## 1. Abstract

Agentic_Serial-killer_Profiler is a two-agent multimodal pipeline that takes up to 15 case photographs, produces a structured forensic report, grounds that report against a local knowledge base, and then writes a behavioral profile of the likely offender. Both agents are derivatives of Qwen3-VL-8B-Thinking and run locally as independent A2A (Agent-to-Agent) services. A Gradio interface drives the workflow, PostgreSQL checkpoints every step so interrupted runs can resume, and a local retrieval (RAG) layer supplies reference material on injury types. The system is a research and educational prototype; its outputs are not forensic findings.

## 2. Objectives

1. Convert case images into a forensic report.
2. Send forensic report to Profiler model via A2A protocol.
3. Use the forensic findings, the images and criminological typology material to produce a behavioral profile in a fixed five-section format.
4. Keep the two reasoning roles separate (Forensic does not profile, Profiler does not derive cause of death).
5. Establish educational reference summaries of injury types listed by forensic report drawn from a curated forensic knowledge base via RAG.
6. Make long, GPU-heavy runs resumable and inspectable.
7. Combine forensic report, profiler analysis and reference summaries into a full final report.

## 3. System Overview

```
Gradio UI (main.py, port 7860)
   |  up to 15 images + optional context
   v
PipelineRun --- case id = SHA-256(images + context)[:12] --- PostgreSQL checkpoints
   |
   |-- 1. Forensic agent (A2A, port 8002) --> forensic report ending in '## Evidence of Trauma (EOT)'
   |
   '-- 2. Run concurrently
         |-- RAG step: read the EOT section, pick reference files (FAISS + Octen embeddings)
         |-- Reference summaries: Forensic agent in reference-summary mode
         '-- Profiler agent (A2A, port 8011): images + forensic report (+ MCP image viewer)
   |
   v
compose_final_report --> pipeline_result.md and 'final' row in Postgres
```

| Component | File(s) | Role |
| --- | --- | --- |
| Orchestrator and UI | `main.py` | Gradio app, subprocess management for both agent servers, the `PipelineRun` state machine |
| Agent servers | `a2a_server.py` | Serves one agent over A2A using `fasta2a` and `pydantic-ai`, on its own port |
| Agent client | `a2a_client.py` | Streaming JSON-RPC (`message/stream`) client; yields text deltas |
| Model runtime | `qwen_agents/model_utils.py` | `QwenVLModel`: local Qwen3-VL inference wrapped as a pydantic-ai `FunctionModel` |
| Forensic agent | `qwen_agents/Forensic_agent/` | System prompt, structured output contract, no tools |
| Profiler agent | `qwen_agents/Profiler_agent/` | System prompt, skill directory, MCP image-viewer toolset |
| Retrieval | `octen_rag.py` | Chunking, FAISS index, trauma-type routing |
| Persistence | `db.py`, `pipeline_state.py` | Postgres checkpoint store, resume and continuation logic |
| Reference choice | `reference_version.py` | Basic, Advanced or Auto reference mode |
| Deployment | `setup_env.sh`, `setup_postgres.sh`, `start.sh`, `backup_postgres.sh` | RunPod bootstrap, database setup and backups |

## 4. Models

The GitHub repository contains only code; the model weights live in the Hugging Face repository's `hf_cache/` directory (roughly 19 GB of weights in total). This section documents them.

### 4.1 Model inventory

| Role | Model | Form | Approx. size |
| --- | --- | --- | --- |
| Forensic agent | [`mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12`](https://huggingface.co/mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12) | LoRA adapter (SFT) over the abliterated Qwen3-VL model | 102.8 MB adapter |
| Profiler agent | [`Kizzington/Qwen3-VL-8B-Thinking-heretic`](https://huggingface.co/Kizzington/Qwen3-VL-8B-Thinking-heretic) | Full 100% abliterated Qwen3-VL model, 4 safetensors shards | about 17.5 GB (5.0 + 4.9 + 4.9 + 2.7) |
| Processor / tokenizer | [`Qwen/Qwen3-VL-8B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | Config, tokenizer and preprocessor files only (no weights) from the base Qwen3-VL model | a few MB |
| Retrieval embeddings | [`Octen/Octen-Embedding-0.6B`](https://huggingface.co/Octen/Octen-Embedding-0.6B) | Embedding model | 1.2 GB |

Both agents therefore share one base model. The Forensic agent is that base plus a fine-tuned adapter; the Profiler is the base plus the the skill directory.

### 4.2 Forensic agent: fine-tuned adapter

The Forensic model is a LoRA adapter named [`Forensic-Agent-4.0-2026-09-08_14.31.12`](https://huggingface.co/mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12) (the name carries a training timestamp of 8 September 2026). Its adapter configuration and model card give the following details.

| Property | Value |
| --- | --- |
| Base model | [`Kizzington/Qwen3-VL-8B-Thinking-heretic`](https://huggingface.co/Kizzington/Qwen3-VL-8B-Thinking-heretic) |
| Method | Supervised fine-tuning (SFT) with TRL 0.24.0, Unsloth-patched |
| LoRA rank / alpha / dropout | r = 8, alpha = 16, dropout = 0 |
| Variants | rsLoRA enabled; DoRA and QLoRA off; bias none |
| Targets | Attention and MLP projections (q/k/v/o, gate/up/down, qkv, linear\_fc1/fc2) in both the vision and language parts, selected by regex |
| Library versions | PEFT 0.20.0, Transformers 5.5.0, PyTorch 2.11.0+cu128, Datasets 4.3.0, Tokenizers 0.22.2 |
| Experiment tracking | Weights & Biases project `Forensic-Agent-4.0` |

At load time `model_utils.py` detects `adapter_config.json`, downloads the named base model, attaches the adapter with PEFT and, unless 4-bit loading is requested, merges it into the base weights (`merge_and_unload`) for faster inference. Two adapter revisions are present in the cached Hugging Face snapshot directory.

It was trained on an abliterated variant of [`Qwen/Qwen3-VL-8B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) called [`Kizzington/Qwen3-VL-8B-Thinking-heretic`](https://huggingface.co/Kizzington/Qwen3-VL-8B-Thinking-heretic) with the dataset manually crafted from scouring the internet, totalling 33 records; 28 for training and 5 for evaluation. Case photographs and most importantly, autopsy reports are very sparse on the internet and with good reason for that. After tweaking the hyperparamaters to the fullest potential(via tool called [`Weights & Biases` now called `CoreWeave Forge`](http://id.coreweave.com/login?next=%2Fwandb%3Foverview)), the model, unfortunately, could only memorize from the training dataset rather than generalizing it. In future, dataset comprising approximately 1,500–2,000 JSONL records worth of aforemetioned case photographs and autopsy reports would be required to achieve the desired outcome.

### 4.3 Profiler agent: base model

The Profiler uses [`Kizzington/Qwen3-VL-8B-Thinking-heretic`](https://huggingface.co/Kizzington/Qwen3-VL-8B-Thinking-heretic) without further training in this project. Its model card describes it as a version of Qwen3-VL-8B-Thinking modified with the Heretic tool (v1.0.1) to remove refusal behavior (abliteration). The card reports a KL divergence of 0.16 from the original model and 0 refusals in 100 test prompts, against 45 for the original. The license is Apache-2.0. This variant was chosen for the purpose of inspecting the crime scene photographs, via skill directory called criminal-behaviour-analysis, that are inherently graphic in nature; the ethical implications are discussed in section 10.

### 4.4 Embedding model

[`Octen/Octen-Embedding-0.6B`](https://huggingface.co/Octen/Octen-Embedding-0.6B) is a text-embedding model fine-tuned with LoRA from [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). It produces 1024-dimension vectors, accepts up to 32,768 tokens, and is Apache-2.0 licensed. The pipeline runs it on CPU through LangChain's `HuggingFaceEmbeddings`. The model was ranked, at the time of designing, in the TOP 10 on the RTEB Leaderboard.


### 4.5 Inference runtime

`QwenVLModel` turns a Hugging Face model into a pydantic-ai `FunctionModel`, so agents can be defined with the standard `Agent` class while inference stays fully local.

| Setting | Forensic agent | Profiler agent |
| --- | --- | --- |
| Temperature | 0.15 (class default) | 0.3 |
| Max new tokens | 6,144 | 12,288 |
| Output-validation retries | 2 | 4 |
| Tools | none | `list_sections`, `view_section`, `view_image` |
| Extra context at first turn | none | 5 'Basic' reference images |
| 4-bit option | `FORENSIC_LOAD_IN_4BIT` | `PROFILER_LOAD_IN_4BIT` |

Common behavior: bfloat16 weights, `sdpa` attention by default, repetition penalty 1.15, token streaming through `TextIteratorStreamer`, and a 300-second stall timeout per chunk. A stream parser hides the model's `<think>...</think>` reasoning and emits only the final answer. Tool calls are expressed as `<tool_call>{json}</tool_call>` blocks and converted into structured tool calls. If generation fails or the token budget is exhausted inside the reasoning block, the runtime emits an explicit `FAILURE:` message instead of an empty result.

A rough planning figure: each 8B model in bfloat16 needs about 17.5 GB for weights alone, so running both agents on one GPU needs roughly 35 GB before activations. For that reason `main.py` assigns the Forensic agent to the first visible GPU and the Profiler to the second when two or more GPUs exist (overridable with `FORENSIC_GPU`, `PROFILER_GPU` and `ASP_MULTI_GPU`).

## 5. Agent Design

### 5.1 Forensic agent

The agent receives only photographs and optional context, with no written case file. Its system prompt requires it to reason inside a think block, then write a forensic-style report with these parts in order: brief summary, examination, cause and manner of death, a `## Murder Weapon (MW)` section, circumstances and external examination, and a mandatory final `## Evidence of Trauma (EOT)` section.

The EOT section is a contract with the downstream pipeline. Its first word must be `Yes` or `No`. If `Yes`, it gives the number of distinct injuries and one numbered entry per injury with Type (Gunshot, Sharp force, Blunt force, Asphyxia, Thermal or Other), Finding, Location and Description. The prompt forbids naming injuries that were not seen and forbids giving secondary effects their own entry.

The same agent has a second mode, reference-summary mode, triggered when a message says so and contains a `<reference_document>` block. In that mode it writes an educational summary using only the supplied document and omits all case conclusions.

### 5.2 Profiler agent

The Profiler is told it is not deriving forensic conclusions; it treats the Forensic report as evidence and writes a different, behavioral document. Its prompt embeds the full `SKILL.md` workflow and `OUTPUT.md` format, and ends with an anti-speculation rule: every claim must rest on something visible in the images or stated in the Forensic report, and weak evidence must be called weak rather than filled with invented specifics.

Required output sections, in order:

1. Crime Scene Report
2. Victimology
3. Organization Level (Organized, Disorganized or Mixed)
4. Offender Typology (drawing on the Palermo/Mastronardi and Holmes & Holmes frameworks)
5. Rationale & Confidence (explicit Low, Medium or High)

An output validator logs a warning if the user chose the Advanced reference mode but the model never opened an Advanced page.

## 6. Retrieval (RAG) Subsystem

**Corpus.** Seven markdown files in `forensic_knowledge/` totalling about 406 KB: asphyxiation, blunt force trauma, gunshot wounds, postmortem changes, sharp force injury, sudden natural death and thermal injuries.

**Indexing.** Files are split with a recursive character splitter (800-character chunks, 100 overlap) and embedded with Octen-Embedding-0.6B into a FAISS index stored in `.octen_faiss_index/`. By default the index is deleted at server start (`RAG_REBUILD_ON_STARTUP`) and rebuilt on first use. Queries retrieve the top 5 chunks, scored as 1 / (1 + distance).

**Trigger logic.** Retrieval is conditional, not automatic:

1. Extract the EOT section from the Forensic report by heading match.
2. If the section's first word is `Yes`, run retrieval; if `No`, skip it and report that no reference summaries apply.
3. Map the `Type:` labels in the EOT list to trauma categories and then to specific knowledge files.
4. If there is no EOT section, fall back to a keyword scan over the whole report, ignoring matches preceded within 40 characters by a negation cue such as 'no', 'without' or 'ruled out'.

**Reference summaries.** For each selected file, the complete text is loaded into the Forensic agent in reference-summary mode, which writes a short educational summary (definitions, mechanism, characteristic features, look-alikes, caveats). Summaries are generated one file at a time and are labelled in the final report. An optional `RAG_CHARS` limit truncates long files and tells the model to say so.

## 7. Pipeline Orchestration and Fault Tolerance

`PipelineRun` runs four stages as nested async generators so the UI streams partial text live: load saved progress, run the Forensic agent, run RAG, reference summaries and Profiler concurrently, then compose the final report.

**Case identity.** The case id is the first 12 hex characters of a SHA-256 over the image file contents (or URLs, in order) plus the extra context. Re-uploading the same images in the same order finds the same case.

**Checkpointing.** Each step (`meta`, `forensic`, `rag`, `education`, `profiler`, `profiler_version`, `final`) is stored in a `pipeline_steps` table keyed by case id and step, with status `partial` or `complete`. Streaming output is flushed to the database at least every 10 seconds.

**Resume modes.** *Continue* re-prompts the model with the saved partial text (trimmed to the last full line) and asks it to continue without repeating; *As-is* accepts the partial text as complete; *Fresh* deletes saved progress. Saved cases can also be chosen from a dropdown.

**Selective re-runs.** If the Forensic report is reused, downstream steps are reused too. Changing the Profiler reference version re-runs only the Profiler. Invalidating the Forensic step invalidates everything downstream.

**Output guarding.** Profiler output that is empty, begins with `FAILURE:`, or is a raw tool-call block is rejected, saved under `profiler_failed`, and discarded instead of being presented as a profile.

**Process management.** Agent servers are subprocesses with per-agent log files (`server_logs/`), exit-code logging, and automatic single-agent restart if a port is found closed mid-run. The database layer degrades gracefully: if Postgres is unreachable the pipeline keeps running and the UI states that progress is not being saved.

## 8. Profiler Reference Material and MCP Tooling

The Profiler's knowledge of criminological typologies comes from reference images packaged as a skill, `criminal-behavioral-analysis`:

- **Basic version:** 5 images under `assets/short/`, always attached to the first turn (downscaled to 1,600 px on the long side). They cover the Palermo/Mastronardi typology, the Holmes & Holmes typology and the Organized/Disorganized/Mixed classification.
- **Advanced version:** 23 images under `assets/long/`, grouped into 6 sections (introduction, study parameters and results, serial-murder investigations, body-disposal pathways, offenders who used the same disposal scenarios, offenders who changed pathways).

The Advanced pages are exposed by a FastMCP stdio server (`image_viewer_mcp.py`) with three tools: `list_sections` (sections plus a description of each image), `view_section` (up to 4 images per call, with an offset) and `view_image`. The reference-version dropdown works through a marker line in the prompt: **Basic** removes the tools for that run (enforced in code), **Advanced** instructs the model to consult relevant sections (an instruction only; the code can log but not force it), and **Auto** leaves the choice to the model.

## 9. Data, Storage and Deployment

- **Runtime:** Python 3.13, managed with `uv` and a frozen lock file. Key dependencies: torch, transformers (5.1 to 5.5), peft, bitsandbytes, accelerate, pydantic-ai-slim, fasta2a 2.0.1, fastmcp, gradio 6, asyncpg, FAISS (CPU), LangChain community/huggingface, sentence-transformers.
- **Target platform:** A RunPod GPU pod with a network volume at `/workspace` that holds the model cache (`HF_HOME`), the uv cache and database backups, so a pod restart does not re-download weights.
- **Startup:** `start.sh` sources `setup_env.sh` (installs uv, sets cache and port variables, switches the Hugging Face hub to offline mode once the cache is populated), runs `setup_postgres.sh` (installs PostgreSQL, creates the `asp` database and role, restores the latest backup, starts a backup loop every 300 seconds), runs `uv sync --frozen`, then launches `main.py`. Starting the two agent servers is done from the UI.
- **Ports:** UI 7860, Profiler agent 8011, Forensic agent 8002.
- **Uploads:** Images are copied into `temp_uploads/` and served to the agents as local Gradio file URLs; A gallery lets users reuse earlier uploads.

## 10. GitHub versus Hugging Face Repositories

|  | GitHub `Agentic_Serial-killer_Profiler` | Hugging Face `Agentic_Serial-killer_Profiler` |
| --- | --- | --- |
| Size | about 64 MB | 39.4 GB |
| Application code, skills, knowledge files | Yes | Yes (identical layout) |
| Postgres backup (`pg-backups/`) | Yes | Yes |
| Model weights and caches (`hf_cache/`) | No | Yes: Profiler base (about 17.5 GB), Forensic LoRA adapter (102.8 MB), Octen embeddings (1.2 GB), Qwen processor files, Xet shard cache |

The code refers to models only by Hub id, so the GitHub repository is runnable but downloads weights on first start; with `HF_HOME` pointed at a populated `hf_cache/`, the system runs fully offline.

## 11. Limitations, Risks and Ethics

**Accuracy.** The interface itself warns that outputs may be inaccurate. Vision-language models can misread injuries, and a behavioural profile built on a mistaken forensic report inherits the mistake. The Profiler's output is explicitly framed as the most likely interpretation, not a finding.

**Sparse training data.** The dataset was manually crafted from scouring the internet, totalling 33 records; 28 for training and 5 for evaluation. Case photographs and most importantly, autopsy reports are very sparse on the internet and with good reason for that. In the future, a dataset comprising approximately 1,500–2,000 JSONL records, incorporating the aforementioned forensic photographs and corresponding autopsy reports, may be required to achieve the desired level of generalization when analyzing previously unseen cases.

**Abliterated base model.** Both agents run on a model with refusal behavior deliberately removed. This allows engagement with sensitive imagery but also means the model has weaker built-in safeguards like not fully adhering to the system prompt 100%; deployment should be restricted to controlled use.
 
**Sensitive data.** The Hugging Face repository includes uploaded sample photographs that are inherently graphic in nature since they are mostly whose autopsy images, and a Postgres dump that may contain stored case outputs.

**Method limits.** The Advanced reference mode cannot be forced; the typology frameworks (Palermo/Mastronardi, Holmes & Holmes, organized/disorganized) are contested in the criminology literature and are not validated predictors; the RAG corpus covers five trauma categories plus two general topics.

## 12. Future Work

-  A dataset comprising approximately 1,500–2,000 JSONL records, incorporating the aforementioned forensic photographs and corresponding autopsy reports, may be required to achieve the desired level of generalization when analyzing previously unseen cases.
- Build a small expert-reviewed test set to measure forensic-report accuracy and EOT format compliance.
- Extend the knowledge base to more injury categories and add source citations in the summaries.

## 13. Conclusion

Agentic_Serial-killer_Profiler shows a complete, locally hosted agentic workflow: two specialised Qwen3-VL-8B agents connected by A2A protocol, a conditional retrieval layer(RAG), tool use through MCP, and PostgreSQL database-backed resumability. The project's main strengths are its strict inter-agent output contract and its robust handling of long, failure-prone generations. Its main gaps are the sparse training data and fully abliterated models involved.

## Appendix A: Key Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_HOME` | `/workspace/hf_cache` | Model cache location |
| `HF_HUB_OFFLINE` | auto (1 when cache populated) | Offline mode |
| `HF_TOKEN` | unset; You have to manually set the token | First download of private repos |
| `FORENSIC_MODEL_ID` | `mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12` | Forensic model or adapter |
| `PROFILER_MODEL_ID` | `Kizzington/Qwen3-VL-8B-Thinking-heretic` | Profiler model |
| `OCTEN_MODEL_ID` | `Octen/Octen-Embedding-0.6B` | Embedding model |
| `FORENSIC_AGENT_PORT` / `PROFILER_AGENT_PORT` | 8002 / 8011 | Agent servers |
| `FORENSIC_GPU` / `PROFILER_GPU` / `ASP_MULTI_GPU` | auto / auto / true | GPU placement |
| `FORENSIC_LOAD_IN_4BIT` / `PROFILER_LOAD_IN_4BIT` | false | 4-bit loading |
| `ATTN_IMPLEMENTATION` | `sdpa` | Attention backend |
| `RAG_REBUILD_ON_STARTUP` / `RAG_CHARS` | true / 0 | Index rebuild, file truncation |
| `DATABASE_URL` | local Postgres, database `asp` | Checkpoint store |
| `SERVER_READY_TIMEOUT_SECONDS` | 300 | Wait for agent servers |
