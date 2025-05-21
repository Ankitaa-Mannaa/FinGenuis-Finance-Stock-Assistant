// Get references
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");

// Handle form submit
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const userInput = input.value.trim();
    if (!userInput) return;

    addMessage(userInput, "user-message");
    input.value = "";

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    });

    const data = await response.json();
    addMessage(data.response, "bot-message");
});

// Add message to chatbox
function addMessage(text, className) {
    const msg = document.createElement("div");
    msg.className = `message ${className}`;

    if (className === "user-message") {
        msg.innerHTML = `
            <div class="label"><strong>You:</strong></div>
            <div class="text">${text}</div>
        `;
    } else {
        msg.innerHTML = `
            <div class="label"><strong>🤖 Chatbot:</strong></div>
            <div class="text">${text}</div>
        `;
    }

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;

    // ➡️ If chatbot sent message, check for stock charts
    if (className === "bot-message") {
        handleNewChartPopups();
    }
}

// ➕ Handle stock chart popup functionality
function handleNewChartPopups() {
    const imgs = document.querySelectorAll('.stock-img');

    imgs.forEach(img => {
        const id = img.id.replace('stockImg_', '');
        const modal = document.getElementById('imgModal_' + id);
        const modalImg = document.getElementById('imgInModal_' + id);
        const closeBtn = document.getElementById('closeModal_' + id);

        if (img && modal && modalImg && closeBtn) {
            img.onclick = function() {
                modal.style.display = "block";
                modalImg.src = img.src;
            };
            closeBtn.onclick = function() {
                modal.style.display = "none";
            };
        }
    });
}

// ➕ NEW: Stock Prediction Code
async function predictStock() {
    const symbol = document.getElementById("stock-symbol").value || "AAPL";
    const days = document.getElementById("predict-days").value || 7;

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, days })
    });

    const data = await res.json();

    document.getElementById("prediction-result").innerHTML = `<pre>${data.predictions}</pre>`;
    document.getElementById("prediction-chart").innerHTML = `<img src="${data.chart}" alt="Prediction Chart" />`;
}
