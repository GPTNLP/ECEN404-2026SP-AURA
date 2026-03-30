const chatContainer = document.getElementById("chat-container");
const statusBadge = document.getElementById("status-badge");
const cpuVal = document.getElementById("cpu-val");
const gpuVal = document.getElementById("gpu-val");
const dbVal = document.getElementById("db-val");
const netVal = document.getElementById("net-val");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

let ws = null;
let reconnectTimer = null;

function setEmptyState() {
    if (!chatContainer) return;

    if (chatContainer.children.length === 0) {
        chatContainer.innerHTML = `
            <div class="empty-state">
                <h2>AURA is ready</h2>
                <p>Send a message below to talk to the robot.</p>
            </div>
        `;
    }
}

function clearEmptyState() {
    const empty = chatContainer.querySelector(".empty-state");
    if (empty) empty.remove();
}

function updateStatus(text, stateClass = "ready") {
    if (!statusBadge) return;
    statusBadge.textContent = text;
    statusBadge.className = `status-indicator ${stateClass}`;
}

function updateConnection(text) {
    if (!netVal) return;
    netVal.textContent = text;
}

function updateCpu(value) {
    if (!cpuVal) return;
    cpuVal.textContent = `CPU: ${value}`;
}

function updateGpu(value) {
    if (!gpuVal) return;
    gpuVal.textContent = `GPU: ${value}`;
}

function updateDb(value) {
    if (!dbVal) return;
    dbVal.textContent = value || "--";
}

function appendMessage(sender, text) {
    if (!chatContainer) return;

    clearEmptyState();

    const normalizedSender =
        sender === "user" ? "user" :
        sender === "ai" ? "ai" :
        "system";

    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", normalizedSender);

    const wrapper = document.createElement("div");

    const bubbleDiv = document.createElement("div");
    bubbleDiv.classList.add("bubble");
    bubbleDiv.textContent = text || "";

    const metaDiv = document.createElement("div");
    metaDiv.classList.add("message-meta");
    metaDiv.textContent =
        normalizedSender === "user" ? "User" :
        normalizedSender === "ai" ? "AURA" :
        "System";

    wrapper.appendChild(bubbleDiv);
    wrapper.appendChild(metaDiv);
    msgDiv.appendChild(wrapper);
    chatContainer.appendChild(msgDiv);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function handleMessage(data) {
    if (!data || typeof data !== "object") return;

    if (data.type === "status") {
        const text = String(data.data || "Ready");

        let stateClass = "ready";
        if (text.toLowerCase().includes("processing")) stateClass = "processing";
        else if (text.toLowerCase().includes("offline")) stateClass = "offline";

        updateStatus(text, stateClass);
        return;
    }

    if (data.type === "chat") {
        appendMessage(data.sender, data.text);
        return;
    }

    if (data.type === "telemetry") {
        if (typeof data.cpu_percent === "number") {
            updateCpu(`${Math.round(data.cpu_percent)}%`);
        } else {
            updateCpu("--%");
        }

        if (typeof data.gpu_percent === "number") {
            updateGpu(`${Math.round(data.gpu_percent)}%`);
        } else {
            updateGpu("--%");
        }

        if (typeof data.db_name === "string") {
            updateDb(data.db_name);
        }

        if (typeof data.connection === "string") {
            updateConnection(data.connection);
        }

        return;
    }

    if (data.type === "system") {
        appendMessage("system", data.text || "System event");
    }
}

function sendMessage() {
    const text = (chatInput?.value || "").trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(text);
    chatInput.value = "";
}

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

    ws.onopen = () => {
        updateStatus("Connected", "ready");
        updateConnection("Connected to robot");
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch (err) {
            console.error("Invalid WS message:", err);
        }
    };

    ws.onclose = () => {
        updateStatus("Offline - Reconnecting...", "offline");
        updateConnection("Disconnected");
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connectWebSocket, 2500);
    };

    ws.onerror = () => {
        updateStatus("Connection Error", "offline");
        updateConnection("Connection error");
    };
}

sendBtn?.addEventListener("click", sendMessage);

chatInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();
    }
});

setEmptyState();
updateCpu("--%");
updateGpu("--%");
updateDb("--");
updateConnection("Offline");
updateStatus("Booting...", "processing");
connectWebSocket();