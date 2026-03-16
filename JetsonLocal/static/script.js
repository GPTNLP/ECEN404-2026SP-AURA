const chatContainer = document.getElementById("chat-container");
const statusBadge = document.getElementById("status-badge");
const cpuVal = document.getElementById("cpu-val");
const netVal = document.getElementById("net-val");

let ws = null;
let reconnectTimer = null;

function setEmptyState() {
    if (!chatContainer) return;

    if (chatContainer.children.length === 0) {
        chatContainer.innerHTML = `
            <div class="empty-state">
                <h2>AURA is ready</h2>
                <p>
                    Waiting for voice, keyboard, or system events from the Jetson runtime.
                    Incoming user and AI messages will appear here live.
                </p>
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

function updateNetwork(text) {
    if (!netVal) return;
    netVal.textContent = text;
}

function updateCpu(value) {
    if (!cpuVal) return;
    cpuVal.textContent = value;
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
        if (text.toLowerCase().includes("listening")) stateClass = "listening";
        else if (text.toLowerCase().includes("processing")) stateClass = "processing";
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
        }

        if (typeof data.network === "string") {
            updateNetwork(data.network);
        }
        return;
    }

    if (data.type === "system") {
        appendMessage("system", data.text || "System event");
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

    ws.onopen = () => {
        updateStatus("Connected", "ready");
        updateNetwork("Online");
        appendMessage("system", "WebSocket connected to local Jetson service.");
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
        updateNetwork("Offline");
        appendMessage("system", "Connection lost. Attempting to reconnect...");

        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connectWebSocket, 2500);
    };

    ws.onerror = () => {
        updateStatus("Connection Error", "offline");
        updateNetwork("Offline");
    };
}

setEmptyState();
updateCpu("--%");
updateNetwork("Offline");
updateStatus("Booting...", "processing");
connectWebSocket();