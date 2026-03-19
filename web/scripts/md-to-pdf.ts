/**
 * Convert arbitrary markdown file to PDF with syntax highlighting
 * Usage: npx tsx scripts/md-to-pdf.ts /path/to/input.md [/path/to/output.pdf]
 */

import { readFileSync } from 'fs';
import { basename } from 'path';
import { marked } from 'marked';
import puppeteer from 'puppeteer';
import { codeToHtml } from 'shiki';

const inputPath = process.argv[2];
if (!inputPath) {
  console.error('Usage: npx tsx scripts/md-to-pdf.ts <input.md> [output.pdf]');
  process.exit(1);
}

const outputPath = process.argv[3] || inputPath.replace(/\.md$/, '.pdf');

async function highlightCode(code: string, lang: string): Promise<string> {
  try {
    return await codeToHtml(code, {
      lang: lang || 'text',
      theme: 'github-dark'
    });
  } catch {
    return `<pre style="background:#24292e;color:#e1e4e8;padding:1em;border-radius:5px;overflow-x:auto;"><code>${code}</code></pre>`;
  }
}

async function processMarkdown(markdown: string): Promise<string> {
  // Find all code blocks
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
  const matches: { full: string; lang: string; code: string }[] = [];
  
  let match;
  while ((match = codeBlockRegex.exec(markdown)) !== null) {
    matches.push({
      full: match[0],
      lang: match[1] || 'text',
      code: match[2]
    });
  }
  
  console.log(`Found ${matches.length} code blocks to highlight...`);
  
  // Highlight all code blocks
  const highlightedBlocks = await Promise.all(
    matches.map(async (m) => ({
      full: m.full,
      html: await highlightCode(m.code, m.lang)
    }))
  );
  
  // Replace code blocks with highlighted HTML
  let processedMd = markdown;
  for (const { full, html } of highlightedBlocks) {
    processedMd = processedMd.replace(full, `<div class="code-block">${html}</div>`);
  }
  
  // Convert rest of markdown (no code blocks now)
  const html = await marked(processedMd);
  return html;
}

async function main() {
  console.log(`Reading ${inputPath}...`);
  const markdown = readFileSync(inputPath, 'utf-8');
  
  console.log('Converting to HTML with syntax highlighting...');
  const htmlBody = await processMarkdown(markdown);
  
  const fullHtml = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${basename(inputPath)}</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      line-height: 1.6;
      max-width: 800px;
      margin: 0 auto;
      padding: 2rem;
      color: #1a1a1a;
    }
    h1, h2, h3 { margin-top: 1.5em; }
    .code-block {
      margin: 1em 0;
      border-radius: 5px;
      overflow: hidden;
    }
    .code-block pre {
      margin: 0;
      padding: 1em;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.5;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 1em 0;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 0.5em;
      text-align: left;
    }
    th { background: #f4f4f4; }
    img { max-width: 100%; }
  </style>
</head>
<body>
${htmlBody}
</body>
</html>
  `;
  
  console.log('Launching browser...');
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.setContent(fullHtml, { waitUntil: 'networkidle0' });
  
  console.log(`Generating PDF: ${outputPath}`);
  await page.pdf({
    path: outputPath,
    format: 'A4',
    margin: {
      top: '2cm',
      right: '2cm',
      bottom: '2cm',
      left: '2cm'
    },
    printBackground: true
  });
  
  await browser.close();
  console.log('Done!');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
