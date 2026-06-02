import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/chat',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const createChat = () => api.post('/create');
export const getChatList = () => api.get('/list');
export const uploadFile = (chatId, formData) => api.post(`/${chatId}/upload`, formData, {
    headers: {
        'Content-Type': 'multipart/form-data',
    }
});
export const queryChat = (chatId, query) => api.post(`/${chatId}/query`, { query });

export const queryChatStream = async (chatId, query, onToken) => {
    const response = await fetch(`http://localhost:8000/chat/${chatId}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let result = { answer: "", evaluation: null, context: [] };

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.substring(6));
                    if (data.token) {
                        result.answer += data.token;
                        if (onToken) onToken(result.answer);
                    }
                    if (data.done) {
                        result.evaluation = data.evaluation;
                        result.context = data.context;
                    }
                } catch (e) {
                    console.error("Failed to parse stream chunk", e);
                }
            }
        }
    }
    return result;
};

export const getHistory = (chatId) => api.get(`/${chatId}/history`);

export default api;
