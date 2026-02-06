export function buildPdfHtml(content: string, title: string): string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(title)} — Architecture Document</title>
<style>
	:root {
		--bg: #ffffff;
		--bg-code: #0d1117;
		--bg-code-inline: #f0f1f3;
		--bg-table-header: #f6f8fa;
		--border: #d1d9e0;
		--border-light: #e8ecf0;
		--text-primary: #1f2328;
		--text-secondary: #424a53;
		--text-muted: #6e7781;
		--accent: #1a7f37;
		--blue: #0969da;
		--purple: #8250df;
		--orange: #bf8700;
		--red: #cf222e;
		--radius-sm: 4px;
		--radius-md: 6px;
	}

	* {
		margin: 0;
		padding: 0;
		box-sizing: border-box;
	}

	body {
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
		background: var(--bg);
		color: var(--text-secondary);
		font-size: 14px;
		line-height: 1.7;
		-webkit-print-color-adjust: exact;
		print-color-adjust: exact;
	}

	/* Headings */
	h1 {
		font-size: 25px;
		font-weight: 700;
		color: var(--text-primary);
		margin: 0 0 16px 0;
		padding-bottom: 8px;
		border-bottom: 1px solid var(--border);
	}

	h2 {
		font-size: 20px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 28px 0 12px 0;
		padding-bottom: 6px;
		border-bottom: 1px solid var(--border-light);
		page-break-after: avoid;
	}

	h3 {
		font-size: 17px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 20px 0 8px 0;
		page-break-after: avoid;
	}

	h4 {
		font-size: 15px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 16px 0 6px 0;
		page-break-after: avoid;
	}

	h5, h6 {
		font-size: 14px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 12px 0 4px 0;
		page-break-after: avoid;
	}

	/* Paragraphs & lists */
	p {
		margin: 0 0 10px 0;
	}

	ul, ol {
		margin: 0 0 10px 0;
		padding-left: 24px;
	}

	li {
		margin-bottom: 3px;
	}

	/* Inline code */
	code {
		font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
		background: var(--bg-code-inline);
		padding: 1px 5px;
		border-radius: 3px;
		font-size: 13px;
		color: var(--text-primary);
	}

	/* Code blocks — keep dark background since shiki tokens have light-on-dark colors */
	.code-block-wrapper {
		position: relative;
		margin: 12px 0;
		page-break-inside: avoid;
	}

	.code-lang {
		position: absolute;
		top: 6px;
		right: 8px;
		font-size: 11px;
		color: #6e7681;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-family: -apple-system, BlinkMacSystemFont, sans-serif;
		z-index: 1;
	}

	.arch-pre {
		background: var(--bg-code);
		border: 1px solid #30363d;
		padding: 12px;
		border-radius: var(--radius-md);
		overflow-x: hidden;
		font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
		font-size: 12px;
		white-space: pre-wrap;
		word-wrap: break-word;
		line-height: 1.5;
		color: #e6edf3;
		margin: 0;
	}

	.arch-pre code {
		background: none;
		padding: 0;
		border: none;
		color: inherit;
		font-family: inherit;
		font-size: inherit;
	}

	.shiki-highlighted .line {
		display: block;
	}

	/* Strong & emphasis */
	strong {
		color: var(--text-primary);
		font-weight: 600;
	}

	em {
		font-style: italic;
	}

	/* Blockquotes */
	blockquote {
		margin: 12px 0;
		padding: 8px 16px;
		border-left: 3px solid var(--purple);
		background: #f5f0ff;
		border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
		color: var(--text-muted);
		page-break-inside: avoid;
	}

	/* Tables */
	table {
		width: 100%;
		border-collapse: collapse;
		margin: 12px 0;
		font-size: 13px;
		page-break-inside: avoid;
	}

	th {
		text-align: left;
		padding: 8px 10px;
		background: var(--bg-table-header);
		border: 1px solid var(--border);
		font-weight: 600;
		color: var(--text-primary);
	}

	td {
		padding: 6px 10px;
		border: 1px solid var(--border);
	}

	tr:nth-child(even) td {
		background: #f6f8fa;
	}

	hr {
		border: none;
		border-top: 1px solid var(--border-light);
		margin: 18px 0;
	}

	/* Images / SVGs */
	img {
		max-width: 100%;
		width: auto;
		height: auto;
		border-radius: var(--radius-sm);
		margin: 12px 0;
		display: block;
		object-fit: contain;
		page-break-inside: avoid;
	}

	/* Links */
	a {
		color: var(--blue);
		text-decoration: none;
	}
</style>
</head>
<body>
${content}
</body>
</html>`;
}

export function escapeHtml(text: string): string {
	return text
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;');
}
