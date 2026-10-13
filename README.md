# colgrep-pleias-demo

> **Heads up:** this is set up for my 2019 Intel MacBook Pro, with colgrep on its 4 GB Radeon Pro 5500M and Pleias on the CPU. A question takes several minutes end to end. On a different machine, change the `--device` and `-ngl` flags to fit your hardware.

I ask a question about a repo, `colgrep --agent` digs up the code, and Pleias-RAG-1B reads it and writes an answer. The page shows the raw Pleias output, section markers and all, and every run gets saved as a JSON trace.

## Install

1. **llama.cpp with Vulkan.** On an Intel Mac this goes through MoltenVK. Follow the Vulkan section of llama.cpp's build docs, then build:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp ~/llama-vulkan/llama.cpp
   cd ~/llama-vulkan/llama.cpp
   cmake -B build -DGGML_VULKAN=ON
   cmake --build build --config Release -j
   ./build/bin/llama-server --list-devices
   ```

   The last command should list your GPU as `Vulkan0`.

2. **colgrep 1.8 or newer.** Agent mode came in 1.8, so 1.7 won't work. Don't run `colgrep --install-agent`, because we serve the model ourselves.

   ```bash
   brew install lightonai/tap/colgrep
   colgrep --version
   ```

3. **The two models.** About 5 GB total.

   ```bash
   mkdir -p ~/Documents/colgrep-agent-2B/models && cd ~/Documents/colgrep-agent-2B/models
   curl -fL -o colgrep-agent-2B-Q8_0.gguf https://huggingface.co/lightonai/colgrep-agent-2B-GGUF/resolve/main/colgrep-agent-2B-Q8_0.gguf
   curl -fL -o pleias-rag-1b.gguf https://huggingface.co/PleIAs/Pleias-RAG-1B-gguf/resolve/main/Pleias-RAG-1B.gguf
   ```

4. **Point colgrep at the agent server.** You only need to do this once.

   ```bash
   colgrep settings --agent-endpoint http://127.0.0.1:8080/v1
   ```

5. **Python 3.9+.** That's it. `server.py` only uses the standard library.

## Serve

Run each of these in its own terminal. The first two run from `~/llama-vulkan/llama.cpp`.

```bash
# colgrep agent model on the GPU (port 8080)
./build/bin/llama-server -m ~/Documents/colgrep-agent-2B/models/colgrep-agent-2B-Q8_0.gguf \
  --device Vulkan0 -ngl 99 -c 16384 -b 256 -ub 128 --flash-attn auto --special --port 8080

# Pleias-RAG-1B on the CPU (port 8081)
./build/bin/llama-server -m ~/Documents/colgrep-agent-2B/models/pleias-rag-1b.gguf \
  --device none -ngl 0 -c 4096 -np 1 --special --port 8081

# the page (port 8000), pointed at whatever repo you want to ask about
python3 server.py ~/Documents/ColBERT
```

Then open http://127.0.0.1:8000. Both `curl -s http://127.0.0.1:8080/health` and `curl -s http://127.0.0.1:8081/health` should say `{"status":"ok"}` first.

## Stuff to know

- Keep `--special` on both llama-servers. Without it, colgrep can't read its tool calls and Pleias drops its section markers, and nothing tells you it broke.
- The colgrep model and Pleias don't fit on a 4 GB GPU together, which is why Pleias is on the CPU. The 1B model is F16, so it isn't quantized.
- Everything stops when the Mac restarts, so start all three again. To kill them: `pkill -f "llama-server.*--port 808[01]"; pkill -f "python3 server.py"`.
- The first question in a repo builds the colgrep index. The index goes in `~/Library/Application Support/colgrep/`, not in the repo.
- On my machine it's about 6 min for colgrep and 1 to 2 min for Pleias per question, and only one question runs at a time.
- Every run (done, failed or cancelled) auto-saves to `traces/<date>/` with the question, what colgrep did, and exactly what Pleias read and wrote. The dropdown at the top loads any of them back.
- For fine-tuning, the pair is `pleias.prompt` → `pleias.completion`. I fill in `review.rating` and `review.notes` by hand.
- Some traces have a `reference_answer` that Claude wrote from the repo alone, without seeing the trace, plus the code lines that answer the question. It shows at the bottom of the page.
- `traces/` and `logs/` are gitignored. Traces contain code from whatever repo you ask about.
