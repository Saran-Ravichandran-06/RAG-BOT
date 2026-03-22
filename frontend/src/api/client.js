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
export const getHistory = (chatId) => api.get(`/${chatId}/history`);

export default api;
