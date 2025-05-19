# integration/

This directory contains integration modules that connect GoldenSignalsAI to external foundation models (LLMs, embeddings, vision, etc.) via a unified, agentic interface. The primary file is:

- **external_model_service.py**: Provides adapters and a robust abstraction layer for accessing Anthropic Claude, Meta Llama, Amazon Titan, Cohere, xAI Grok, and more. Supports agentic model selection, fallback, and ensemble strategies. Heavily commented for clarity and extensibility.

## Usage
- Import and instantiate `ExternalModelService` in your application code.
- Configure which provider to use for each task (sentiment, explanation, embeddings, vision) via config or `.env`.
- Call methods like `analyze_sentiment`, `generate_explanation`, `get_embeddings`, or `vision_analysis`.
- Use `ensemble_sentiment` for robust, multi-model aggregation.

## Agentic AI
- The interface is designed for agentic workflows: agents can reason about which model/provider to use, fallback on errors, or ensemble multiple outputs for better decisions.

## Extending
- Add new adapters for additional providers by subclassing `ModelProviderAdapter`.
- Expand agentic logic in `ExternalModelService` as new use cases arise.

---

**This layer ensures you can leverage the latest foundation models without disrupting or replacing your existing custom models and infrastructure.**
