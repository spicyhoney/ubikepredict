import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'YouBike 持續缺車預警',
  description: '使用凍結 LightGBM 模型回放新北市 YouBike 缺車持續風險。',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}
