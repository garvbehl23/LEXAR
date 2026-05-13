export function TypingIndicator() {
  return (
    <div className="flex justify-start px-4 py-2 animate-fade-in">
      <div className="bg-white border border-gray-200 rounded-[18px_18px_18px_6px] px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="block w-2 h-2 rounded-full bg-gray-400 animate-bounce-dot"
              style={{ animationDelay: `${i * 0.16}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
