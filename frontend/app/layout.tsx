import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'YouBike 失衡持續預警',
  description: '使用獨立凍結 LightGBM 模型回放新北市 YouBike 缺車與滿柱持續風險。',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}
