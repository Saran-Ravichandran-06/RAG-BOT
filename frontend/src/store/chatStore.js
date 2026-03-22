import { create } from 'zustand';
import { createChat, getChatList, queryChat, uploadFile } from '../api/client';

const useChatStore = create((set, get) => ({
    chats: [],
    activeChatId: null,
    messages: {}, // { chatId: [messages] }
    isLoading: false,
    uploading: false,

    fetchChats: async () => {
        try {
            const res = await getChatList();
            set({ chats: res.data });
        } catch (error) {
            console.error("Failed to fetch chats", error);
        }
    },

    createNewChat: async () => {
        try {
            const res = await createChat();
            const newChat = res.data;
            set((state) => ({
                chats: [newChat, ...state.chats],
                activeChatId: newChat.chat_id,
                messages: { ...state.messages, [newChat.chat_id]: [] }
            }));
            return newChat.chat_id;
        } catch (error) {
            console.error("Failed to create chat", error);
        }
    },

    setActiveChat: (chatId) => set({ activeChatId: chatId }),

    handleUpload: async (file) => {
        const { activeChatId } = get();
        if (!activeChatId) return;

        set({ uploading: true });
        const formData = new FormData();
        formData.append('file', file);

        try {
            await uploadFile(activeChatId, formData);
            // Optionally add a system message saying "File uploaded"
            const sysMsg = { role: 'system', content: `Uploaded ${file.name}` };
            set((state) => ({
                messages: {
                    ...state.messages,
                    [activeChatId]: [...(state.messages[activeChatId] || []), sysMsg]
                }
            }));
        } catch (error) {
            console.error("Upload failed", error);
        } finally {
            set({ uploading: false });
        }
    },

    sendMessage: async (content) => {
        const { activeChatId } = get();
        if (!activeChatId) return;

        const userMsg = { role: 'user', content };

        // Optimistic update
        set((state) => ({
            messages: {
                ...state.messages,
                [activeChatId]: [...(state.messages[activeChatId] || []), userMsg]
            },
            isLoading: true
        }));

        try {
            const res = await queryChat(activeChatId, content);
            const data = res.data;

            const botMsg = {
                role: 'assistant',
                content: data.answer,
                evaluation: data.evaluation,
                sources: data.context
            };

            set((state) => ({
                messages: {
                    ...state.messages,
                    [activeChatId]: [...(state.messages[activeChatId] || []), botMsg]
                }
            }));

        } catch (error) {
            console.error("Failed to send message", error);
        } finally {
            set({ isLoading: false });
        }
    }
}));

export default useChatStore;
