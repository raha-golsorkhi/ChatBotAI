document.addEventListener("DOMContentLoaded", function () {
    const chatLog = document.getElementById("chat-log");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    // Load intents data from intents.json
    fetch("intents.json")
        .then(response => response.json())
        .then(data => {
            const intents = data.intents;

            function appendMessage(message, isUser) {
                const messageDiv = document.createElement("div");
                messageDiv.classList.add(isUser ? "user-message" : "bot-message");
                messageDiv.innerText = message;
                chatLog.appendChild(messageDiv);
                chatLog.scrollTop = chatLog.scrollHeight;
            }

            function jaccardSimilarity(a, b) {
                const setA = new Set(a.split(" "));
                const setB = new Set(b.split(" "));
                const intersection = new Set([...setA].filter(x => setB.has(x)));
                const union = new Set([...setA, ...setB]);
                return intersection.size / union.size;
            }

            function getBotResponse(userInput) {
                const cleanedInput = userInput.trim().toLowerCase();

                // Find the intent with the highest Jaccard similarity to the user input
                let bestSimilarity = 0;
                let bestIntent = null;
                for (const intent of intents) {
                    for (const pattern of intent.patterns) {
                        const similarity = jaccardSimilarity(cleanedInput, pattern);
                        if (similarity > bestSimilarity) {
                            bestSimilarity = similarity;
                            bestIntent = intent;
                        }
                    }
                }

                // If a matching intent is found, choose a random response from that intent
                if (bestIntent && bestIntent.responses && bestIntent.responses.length > 0) {
                    return "Atlas: " + bestIntent.responses[Math.floor(Math.random() * bestIntent.responses.length)];
                } else {
                    return "Atlas: I apologize, but I don't have the information or capability to answer that question at the moment.";
                }
            }

            function handleUserInput() {
                const userMessage = userInput.value;
                if (userMessage.trim() !== "") {
                    appendMessage("You: " + userMessage, true);

                    // Get bot response based on user input
                    const botResponse = getBotResponse(userMessage);
                    appendMessage(botResponse, false);

                    // Clear user input
                    userInput.value = "";
                }
            }

            // Handle user input when the "Send" button is clicked
            sendButton.addEventListener("click", handleUserInput);

            // Handle user input when the "Enter" key is pressed
            userInput.addEventListener("keyup", function (event) {
                if (event.key === "Enter") {
                    handleUserInput();
                }
            });
        })
        .catch(error => {
            console.error("Error loading intents data:", error);
        });
});
