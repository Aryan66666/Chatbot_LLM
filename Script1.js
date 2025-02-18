function sendMessage() {
    const userInput = document.getElementById('user-input').value;

    if (userInput.trim() === "") {
        return;
    }

    displayMessage(userInput, 'user-message');

    document.getElementById('user-input').value = "";

    setTimeout(async () => {
        const botResponse = await getBotResponse(userInput);
        displayMessage(botResponse, 'bot-message');
    }, 1000);
}

function displayMessage(message, messageType) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', messageType);
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);

    chatBox.scrollTop = chatBox.scrollHeight;
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function getBotResponse(userMessage) {
    const requestBody = { question: userMessage };

    try {
        const response = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        const data = await response.json();

        if (response.ok) {
            console.log(data.result);  

            
            await delay(1000);  

            return data.result;  
        } else {
            const errorData = await response.json();
            console.error('Error:', errorData.error);
            return `Error: ${errorData.error}`;
        }
    } catch (error) {
        console.error('Request failed:', error);
        return 'Request failed! Please try again later.';
    }
}
