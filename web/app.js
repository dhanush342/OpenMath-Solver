const solveBtn = document.getElementById("solveBtn");
const questionEl = document.getElementById("question");
const outputEl = document.getElementById("output");
const statusEl = document.getElementById("status");
const maxTokensEl = document.getElementById("maxTokens");

function setStatus(text, running = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("running", running);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;
    const data = await response.json();
    if (data.status === "ready") {
      setStatus("Ready", false);
    } else if (data.status === "loading") {
      setStatus("Loading model...", true);
    } else if (data.status === "error") {
      setStatus("Model error", false);
      outputEl.textContent = data.detail || "Model failed to load.";
    }
  } catch (error) {
    setStatus("Offline", false);
  }
}

refreshStatus();
setInterval(refreshStatus, 4000);

solveBtn.addEventListener("click", async () => {
  const question = questionEl.value.trim();
  if (!question) {
    outputEl.textContent = "Please enter a problem.";
    return;
  }

  const maxTokens = Number(maxTokensEl.value || 200);

  setStatus("Running...", true);
  solveBtn.disabled = true;
  outputEl.textContent = "";

  try {
    const response = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, max_new_tokens: maxTokens })
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Request failed");
    }

    const data = await response.json();
    outputEl.textContent = data.output || "No output";
    setStatus("Done", false);
  } catch (error) {
    outputEl.textContent = error.message;
    setStatus("Error", false);
  } finally {
    solveBtn.disabled = false;
  }
});
