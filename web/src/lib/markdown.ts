import { Marked } from 'marked';
import { highlightCode } from './highlight.js';

const marked = new Marked();

// Generate slug from heading text for anchor links
function slugify(text: string): string {
	return text
		.toLowerCase()
		.replace(/<[^>]*>/g, '')
		.replace(/[^\w\s-]/g, '')
		.replace(/\s+/g, '-')
		.replace(/-+/g, '-')
		.trim();
}

export interface TocEntry {
	level: number;
	text: string;
	id: string;
}

export async function renderArchDoc(
	markdown: string,
	assetBaseUrl?: string
): Promise<{ html: string; toc: TocEntry[] }> {
	const toc: TocEntry[] = [];

	// First pass: collect code blocks for async highlighting
	const codeBlocks: { lang: string; code: string; placeholder: string }[] = [];
	let blockIdx = 0;

	const preprocessed = markdown.replace(
		/^[ \t]*```(\w*)[ \t]*\n([\s\S]*?)^[ \t]*```[ \t]*$/gm,
		(_match, lang, code) => {
			const placeholder = `<!--CODE_BLOCK_${blockIdx}-->`;
			codeBlocks.push({ lang: lang || '', code: code.trimEnd(), placeholder });
			blockIdx++;
			return placeholder;
		}
	);

	// Highlight all code blocks in parallel
	const highlightedBlocks = await Promise.all(
		codeBlocks.map(async ({ lang, code }) => {
			if (!lang || lang === 'text' || lang === 'plaintext') {
				return escapeHtml(code);
			}
			return await highlightCode(code, lang);
		})
	);

	// Rewrite image paths to use asset API if assetBaseUrl provided
	let processed = preprocessed;
	if (assetBaseUrl) {
		processed = processed.replace(
			/!\[([^\]]*)\]\(\.\/([^)]+)\)/g,
			(_match, alt, path) => `![${alt}](${assetBaseUrl}?path=${encodeURIComponent(path)})`
		);
	}

	// Parse markdown to HTML
	let html = await marked.parse(processed);

	// Replace code block placeholders with highlighted content
	// Note: use split/join instead of replace to avoid $-substitution in replacement strings
	// (shiki output often contains $ characters which trigger special replacement patterns)
	for (let i = 0; i < codeBlocks.length; i++) {
		const { lang, placeholder } = codeBlocks[i];
		const highlighted = highlightedBlocks[i];
		const langLabel = lang ? `<span class="code-lang">${escapeHtml(lang)}</span>` : '';
		const replacement = `<div class="code-block-wrapper">${langLabel}<pre class="arch-pre shiki-highlighted"><code>${highlighted}</code></pre></div>`;
		html = html.split(placeholder).join(replacement);
	}

	// Process headings: add IDs and collect TOC
	html = html.replace(
		/<h([1-6])>([\s\S]*?)<\/h[1-6]>/g,
		(_match, level, text) => {
			const id = slugify(text);
			const lvl = parseInt(level);
			toc.push({ level: lvl, text: text.replace(/<[^>]*>/g, ''), id });
			return `<h${level} id="${id}">${text}</h${level}>`;
		}
	);

	return { html, toc };
}

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;');
}

// In-memory cache for rendered docs
const docCache = new Map<string, { html: string; toc: TocEntry[] }>();

export async function renderArchDocCached(
	key: string,
	markdown: string,
	assetBaseUrl?: string
): Promise<{ html: string; toc: TocEntry[] }> {
	const cached = docCache.get(key);
	if (cached) return cached;

	const result = await renderArchDoc(markdown, assetBaseUrl);
	docCache.set(key, result);
	return result;
}
