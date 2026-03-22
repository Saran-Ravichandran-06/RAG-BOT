import React, { useEffect, useState } from 'react';
import { Plus, ChevronLeft, MessageSquare, X, Trash2 } from 'lucide-react';
import clsx from 'clsx';
import axios from 'axios';

const Sidebar = ({ isOpen, onClose, onSelectChat, onNewChat, currentChatId }) => {
    const [chats, setChats] = useState([]);

    const fetchChats = async () => {
        try {
            const response = await axios.get('http://localhost:8000/chat/list');
            setChats(response.data);
        } catch (error) {
            console.error("Failed to fetch chats:", error);
        }
    };

    useEffect(() => {
        if (isOpen) {
            fetchChats();
        }
    }, [isOpen]);

    const handleDeleteChat = async (e, chatId) => {
        e.stopPropagation();
        if (!confirm("Are you sure you want to delete this chat?")) return;

        try {
            await axios.delete(`http://localhost:8000/chat/${chatId}`);
            setChats(prev => prev.filter(c => c.chat_id !== chatId));
            if (currentChatId === chatId) {
                onNewChat();
            }
        } catch (error) {
            console.error("Failed to delete chat:", error);
        }
    };

    return (
        <div
            className={clsx(
                "fixed top-16 left-0 h-[calc(100vh-64px)] bg-gray-950/80 backdrop-blur-md text-white transition-transform duration-300 ease-in-out z-20 w-[280px] border-r border-gray-800",
                isOpen ? "translate-x-0" : "-translate-x-full"
            )}
        >
            {/* Sidebar Toggle (Close) */}
            <button
                onClick={onClose}
                className={clsx(
                    "fixed top-1/2 left-[280px] transform -translate-y-1/2 z-30 p-2 text-gray-400 hover:text-white transition-all duration-300 bg-black/40 hover:bg-black/60 rounded-r-md border border-l-0 border-gray-800",
                    isOpen ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-full pointer-events-none"
                )}
                aria-label="Close sidebar"
            >
                <ChevronLeft size={24} />
            </button>
            <div className="flex flex-col h-full p-4">
                {/* Header / New Chat */}
                <div className="flex items-center mb-6">
                    <button
                        className="flex-1 flex items-center gap-2 px-4 py-2 bg-gray-900 hover:bg-gray-800 rounded-md transition-colors text-sm font-medium border border-gray-700 text-gray-200"
                        onClick={onNewChat}
                    >
                        New Chat
                    </button>
                </div>

                {/* History List */}
                <div className="flex-1 overflow-y-auto">
                    <div className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider pl-1">
                        Recent
                    </div>
                    <div className="space-y-1">
                        {chats.map((chat) => (
                            <div
                                key={chat.chat_id}
                                onClick={() => onSelectChat(chat.chat_id)}
                                className={clsx(
                                    "group flex items-center gap-3 w-full px-3 py-2 text-sm rounded-md transition-colors cursor-pointer",
                                    currentChatId === chat.chat_id
                                        ? "bg-gray-800 text-white"
                                        : "text-gray-400 hover:bg-gray-900 hover:text-gray-200"
                                )}
                            >
                                <MessageSquare size={14} className="flex-shrink-0" />
                                <span className="truncate flex-1 text-left">{chat.title}</span>

                                <button
                                    onClick={(e) => handleDeleteChat(e, chat.chat_id)}
                                    className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
                                    title="Delete Chat"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
