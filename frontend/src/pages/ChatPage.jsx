import React, { useState, useRef, useEffect } from 'react';
import { Plus, Globe, ArrowUp, ChevronRight } from 'lucide-react';
import clsx from 'clsx';
import axios from 'axios';
import Sidebar from '../components/Sidebar';
import MessageBubble from '../components/MessageBubble';
import AnoAI from '../components/ui/animated-shader-background';

const ChatPage = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState("");
    const [chatId, setChatId] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isUrlMode, setIsUrlMode] = useState(false);

    // Refs for file inputs
    const fileInputRef = useRef(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // --- Actions ---

    const handleNewChat = () => {
        setChatId(null);
        setMessages([]);
        setInputValue("");
        setIsUrlMode(false);
        setIsSidebarOpen(false); // Optional: keep sidebar request logic
    };

    const handleSelectChat = async (id) => {
        setChatId(id);
        setIsUrlMode(false);
        setIsLoading(true); // Show loading feedback
        try {
            const response = await axios.get(`http://localhost:8000/chat/${id}/history`);
            setMessages(response.data);
            setIsSidebarOpen(false);
        } catch (error) {
            console.error("Failed to load chat history:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const type = file.name.split('.').pop().toUpperCase();

        // Create chat if not exists
        let currentId = chatId;
        if (!currentId) {
            try {
                const res = await axios.post('http://localhost:8000/chat/create');
                currentId = res.data.chat_id;
                setChatId(currentId);
            } catch (error) {
                console.error("Failed to create chat:", error);
                return;
            }
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Optimistic update? Maybe just a system message
            setMessages(prev => [...prev, { role: 'user', content: `Uploaded file: ${file.name}` }]); // Visual feedback

            await axios.post(`http://localhost:8000/chat/${currentId}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            // Allow backend to process. In a real app we might wait or show progress.
            setMessages(prev => [...prev, { role: 'assistant', content: `Processed ${type} file: ${file.name}` }]);
        } catch (error) {
            console.error("Upload failed:", error);
            setMessages(prev => [...prev, { role: 'assistant', content: `Failed to upload file.` }]);
        }

        // Reset input
        event.target.value = '';
    };

    const handleUrlSubmit = async () => {
        if (!inputValue.trim()) return;

        // Create chat if needed
        let currentId = chatId;
        if (!currentId) {
            try {
                const res = await axios.post('http://localhost:8000/chat/create');
                currentId = res.data.chat_id;
                setChatId(currentId);
            } catch (error) {
                console.error("Failed to create chat:", error);
                return;
            }
        }

        const url = inputValue.trim();
        setInputValue("");
        setIsUrlMode(false); // Exit URL mode?

        setMessages(prev => [...prev, { role: 'user', content: `Ingest URL: ${url}` }]);

        const formData = new FormData();
        formData.append('url', url);

        try {
            await axios.post(`http://localhost:8000/chat/${currentId}/upload`, formData);
            setMessages(prev => [...prev, { role: 'assistant', content: `Successfully ingested URL content.` }]);
        } catch (error) {
            console.error("URL ingest failed:", error);
            setMessages(prev => [...prev, { role: 'assistant', content: `Failed to ingest URL.` }]);
        }
    };

    const handleSendMessage = async () => {
        if (!inputValue.trim()) return;

        if (isUrlMode) {
            handleUrlSubmit();
            return;
        }

        const text = inputValue;
        setInputValue("");

        // Optimistic UI
        setMessages(prev => [...prev, { role: 'user', content: text }]);
        setIsLoading(true);

        // Create chat if needed
        let currentId = chatId;
        if (!currentId) {
            try {
                const res = await axios.post('http://localhost:8000/chat/create');
                currentId = res.data.chat_id;
                setChatId(currentId);
            } catch (error) {
                console.error("Failed to create chat:", error);
                setIsLoading(false);
                return;
            }
        }

        // Add empty assistant bubble
        const tempId = Date.now();
        setMessages(prev => [...prev, {
            id: tempId,
            role: 'assistant',
            content: '',
            sources: [],
            evaluation: null
        }]);

        try {
            const response = await fetch(`http://localhost:8000/chat/${currentId}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

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
                                setMessages(prev => prev.map(msg => 
                                    msg.id === tempId ? { ...msg, content: msg.content + data.token } : msg
                                ));
                            }
                            if (data.done) {
                                setMessages(prev => prev.map(msg => 
                                    msg.id === tempId ? { 
                                        ...msg, 
                                        evaluation: data.evaluation,
                                        sources: data.context
                                    } : msg
                                ));
                            }
                        } catch (e) {
                            console.error("Failed to parse SSE", e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Failed to send message:", error);
            setMessages(prev => prev.map(msg => 
                msg.id === tempId ? { ...msg, content: "Sorry, I encountered an error." } : msg
            ));
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const handleWorldClick = () => {
        if (!isUrlMode) {
            window.open('https://google.com', '_blank');
            setIsUrlMode(true);
            setInputValue(""); // Clear any existing text
        } else {
            setIsUrlMode(false);
            setInputValue(""); // Reset if toggled off
        }
    };

    return (
        <div className="relative w-full h-screen bg-transparent text-gray-100 overflow-hidden flex">
            <AnoAI />

            {/* Hidden Inputs */}
            <input
                type="file"
                ref={fileInputRef}
                accept=".pdf,.txt"
                className="hidden"
                onChange={handleFileUpload}
            />

            {/* Sidebar */}
            <Sidebar
                isOpen={isSidebarOpen}
                onClose={() => setIsSidebarOpen(false)}
                onSelectChat={handleSelectChat}
                onNewChat={handleNewChat}
                currentChatId={chatId}
            />

            {/* Main Content */}
            <div className="flex-1 flex flex-col relative h-full transition-all duration-300">

                {/* Branding Title */}
                <div className="absolute top-6 left-6 z-30 pointer-events-none">
                    <h1 className="text-xl font-bold tracking-tight text-white">RAG BOT</h1>
                </div>

                {/* Sidebar Toggle (Open) */}
                <button
                    onClick={() => setIsSidebarOpen(true)}
                    className={clsx(
                        "fixed top-1/2 left-0 transform -translate-y-1/2 z-30 p-2 text-gray-400 hover:text-white transition-all duration-300 bg-black/40 hover:bg-black/60 rounded-r-md border border-l-0 border-gray-800",
                        isSidebarOpen ? "opacity-0 -translate-x-full pointer-events-none" : "opacity-100 translate-x-0"
                    )}
                    aria-label="Open sidebar"
                >
                    <ChevronRight size={24} />
                </button>

                {/* Chat History Area */}
                {messages.length > 0 && (
                    <div className="flex-1 overflow-y-auto w-full custom-scrollbar">
                        <div className="max-w-3xl mx-auto w-full pb-32 pt-10">
                            {messages.map((msg, idx) => (
                                <MessageBubble key={idx} message={msg} />
                            ))}
                            <div ref={messagesEndRef} />
                        </div>
                    </div>
                )}

                {/* Input Area */}
                <div
                    className={clsx(
                        "transition-all duration-500 ease-in-out w-full flex justify-center px-4",
                        messages.length === 0
                            ? "h-full items-center"
                            : "absolute bottom-0 left-0 pb-3 pt-5"
                    )}
                >
                    <div className="w-full max-w-4xl relative">
                        {/* Title if empty */}
                        {messages.length === 0 && (
                            <div className="text-center mb-8 text-gray-200">
                                {isLoading ? (
                                    <div className="animate-pulse flex flex-col items-center">
                                        <div className="h-8 w-64 bg-gray-800 rounded mb-4"></div>
                                        <div className="h-4 w-48 bg-gray-800 rounded"></div>
                                    </div>
                                ) : (
                                    <h1 className="text-4xl font-bold">
                                        {isUrlMode ? "Enter Website URL" : "How can I help you today?"}
                                    </h1>
                                )}
                            </div>
                        )}

                        {/* Input Box */}
                        <div className={clsx(
                            "bg-white rounded-[22px] p-1.5 pl-3 pr-1.5 flex items-end gap-3 shadow-lg border transition-colors",
                            isUrlMode ? "border-blue-500/50" : "border-gray-300"
                        )}>

                            {/* Left Icons */}
                            <div className="flex items-center gap-2 pb-2 text-gray-700">
                                <button
                                    onClick={() => fileInputRef.current?.click()}
                                    className="p-1.5 hover:bg-gray-200 rounded-full transition-colors hover:text-black"
                                    title="Upload File (PDF or Text)"
                                >
                                    <Plus size={20} />
                                </button>
                                <button
                                    onClick={handleWorldClick}
                                    className={clsx(
                                        "p-1.5 rounded-full transition-colors hover:text-black",
                                        isUrlMode ? "text-blue-600 bg-blue-100" : "hover:bg-gray-200"
                                    )}
                                    title="Browse Website"
                                >
                                    <Globe size={18} />
                                </button>
                            </div>

                            {/* Text Input */}
                            <textarea
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={isUrlMode ? "Paste website URL here..." : "Chat with RAG-BOT"}
                                className="flex-1 bg-transparent border-0 outline-none text-black placeholder-gray-500 resize-none py-1 max-h-[200px] min-h-[30px]"
                                rows={1}
                                style={{ height: 'auto', minHeight: '30px' }}
                            />

                            <button
                                onClick={handleSendMessage}
                                disabled={!inputValue.trim() || isLoading}
                                className={clsx(
                                    "p-2 rounded-full mb-1 transition-all duration-200",
                                    inputValue.trim() && !isLoading
                                        ? "bg-black text-white hover:bg-gray-800"
                                        : "bg-gray-300 text-gray-500 cursor-not-allowed"
                                )}
                            >
                                <ArrowUp size={20} strokeWidth={2.5} />
                            </button>
                        </div>

                        {/* Footer disclaimer */}
                        <div className="text-center mt-2 text-[10px] text-gray-500">
                            Create with <span className="font-semibold text-gray-400">RAG Bot</span>. content may be inaccurate.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatPage;
