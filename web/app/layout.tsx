import type { Metadata } from "next";
import "./globals.css";
import { ChatProvider } from "@/context/ChatContext";

export const metadata: Metadata = {
  title: "LEXAR — Indian Legal AI",
  description: "Evidence-grounded answers to Indian legal questions.",
  icons: { icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚖️</text></svg>" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ChatProvider>{children}</ChatProvider>
      </body>
    </html>
  );
}
