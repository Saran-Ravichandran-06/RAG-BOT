import React from 'react';
import ReactMarkdown from 'react-markdown';
import clsx from 'clsx';

const MessageBubble = ({ message }) => {
    const isBot = message.role === 'assistant';

    return (
        <div className={clsx(
            "flex gap-4 w-full p-6 mx-auto max-w-4xl",
            isBot ? "bg-transparent justify-start" : "bg-transparent justify-end flex-row-reverse"
        )}>
            <div className={clsx(
                "flex-1 min-w-0 max-w-[99%]",
                !isBot && "text-right"
            )}>
                <div className={clsx(
                    "prose prose-invert max-w-none text-sm leading-relaxed rounded-2xl px-5 py-3 shadow-sm inline-block text-left",
                    "bg-indigo-600/20 text-indigo-100"
                )}>
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>

                {/* Evaluation badge removed */}
            </div>
        </div>
    );
};

export default MessageBubble;
