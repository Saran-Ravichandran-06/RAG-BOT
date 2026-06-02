import React from 'react';
import ReactMarkdown from 'react-markdown';
import clsx from 'clsx';

const MessageBubble = ({ message }) => {
    const isBot = message.role === 'assistant';

    return (
        <div className={clsx(
            "flex gap-4 w-full p-4 mx-auto",
            isBot ? "justify-start" : "justify-end"
        )}>
            <div className={clsx(
                "min-w-0",
                isBot ? "max-w-3xl" : "max-w-xl"
            )}>
                <div className={clsx(
                    "prose prose-invert max-w-none text-sm leading-relaxed rounded-2xl px-5 py-3 shadow-sm inline-block text-left backdrop-blur-md border",
                    isBot 
                      ? "bg-white/10 border-white/10 text-gray-100" 
                      : "bg-blue-500/20 border-blue-500/30 text-white"
                )}>
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
            </div>
        </div>
    );
};

export default MessageBubble;
