/**
 * Pre-generate all architecture docs (HTML + PDF) for instant serving.
 *
 * Usage:  npx tsx scripts/generate-docs.ts [--html-only] [--pdf-only] [slug1 slug2 ...]
 *
 * Output: ../data/generated-docs/{slug}/doc.json  (html, toc, title, markdown)
 *         ../data/generated-docs/{slug}/doc.pdf
 */

import { readdirSync, readFileSync, mkdirSync, writeFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { renderArchDoc } from '../src/lib/markdown.js';
import { buildPdfHtml } from '../src/lib/pdf-html.js';
import puppeteer, { type Browser } from 'puppeteer';

const DATA_DIR = resolve(import.meta.dirname!, '../../data');
const ARCH_DOCS_DIR = resolve(DATA_DIR, 'architecture-docs');
const OUTPUT_DIR = resolve(DATA_DIR, 'generated-docs');

// ---- CLI args ----
const args = process.argv.slice(2);
const htmlOnly = args.includes('--html-only');
const pdfOnly = args.includes('--pdf-only');
const slugArgs = args.filter((a) => !a.startsWith('--'));

// ---- Collect slugs ----
function getSlugs(): string[] {
	if (slugArgs.length > 0) return slugArgs;
	return readdirSync(ARCH_DOCS_DIR, { withFileTypes: true })
		.filter((d) => d.isDirectory())
		.map((d) => d.name)
		.sort();
}

// ---- Rewrite web HTML img URLs to inline base64 for PDF ----
function inlineImagesFromWebHtml(html: string, docDir: string): string {
	// Web HTML has: <img src="/api/project/{slug}/architecture-doc/asset?path=..." alt="...">
	return html.replace(
		/<img\s+src="\/api\/project\/[^/]+\/architecture-doc\/asset\?path=([^"]+)"\s+alt="([^"]*)"\s*\/?>/g,
		(_match, encodedPath, alt) => {
			const assetPath = decodeURIComponent(encodedPath);
			const fullPath = resolve(docDir, assetPath);

			if (!fullPath.startsWith(docDir)) {
				return `<p><em>[Image not available: ${alt}]</em></p>`;
			}

			try {
				const content = readFileSync(fullPath);
				const ext = assetPath.split('.').pop()?.toLowerCase();

				if (ext === 'svg') {
					const base64 = content.toString('base64');
					return `<img src="data:image/svg+xml;base64,${base64}" alt="${alt}" />`;
				} else if (ext === 'png') {
					const base64 = content.toString('base64');
					return `<img src="data:image/png;base64,${base64}" alt="${alt}" />`;
				}
				return `<p><em>[Unsupported image format: ${alt}]</em></p>`;
			} catch {
				return `<p><em>[Image not found: ${alt}]</em></p>`;
			}
		}
	);
}

// ---- Main ----
async function main() {
	const slugs = getSlugs();
	const total = slugs.length;
	console.log(`Generating docs for ${total} projects...`);

	mkdirSync(OUTPUT_DIR, { recursive: true });

	let browser: Browser | null = null;
	if (!htmlOnly) {
		console.log('Launching Puppeteer browser...');
		browser = await puppeteer.launch({
			headless: true,
			protocolTimeout: 600_000, // 10 minutes - critical for massive docs like build-os
			args: [
				'--no-sandbox', 
				'--disable-setuid-sandbox', 
				'--disable-dev-shm-usage',
				'--disable-gpu',
				'--no-zygote',
				'--single-process'
			]
		});
	}

	const errors: { slug: string; phase: string; error: string }[] = [];
	const startTime = Date.now();

	for (let i = 0; i < total; i++) {
		const slug = slugs[i];
		const docPath = resolve(ARCH_DOCS_DIR, slug, 'index.md');
		const docDir = dirname(docPath);
		const outDir = resolve(OUTPUT_DIR, slug);

		const prefix = `[${i + 1}/${total}] ${slug}`;

		if (!existsSync(docPath)) {
			console.log(`${prefix} — SKIP (no index.md)`);
			continue;
		}

		mkdirSync(outDir, { recursive: true });

		const markdown = readFileSync(docPath, 'utf-8');
		const titleMatch = markdown.match(/^#\s+(.+)$/m);
		const title = titleMatch ? titleMatch[1] : slug;

		// ---- HTML + TOC ----
		let webHtml: string | null = null;
		if (!pdfOnly) {
			try {
				const webAssetBase = `/api/project/${slug}/architecture-doc/asset`;
				const { html, toc } = await renderArchDoc(markdown, webAssetBase);
				webHtml = html;

				const docJson = JSON.stringify({ html, toc, title, markdown });
				writeFileSync(resolve(outDir, 'doc.json'), docJson, 'utf-8');

				process.stdout.write(`${prefix} — HTML OK`);
			} catch (e: any) {
				console.log(`${prefix} — HTML FAILED: ${e.message}`);
				errors.push({ slug, phase: 'html', error: e.message });
				continue; // skip PDF too if HTML fails
			}
		}

		// ---- PDF ----
		if (!htmlOnly && browser) {
			try {
				// Reuse the web HTML — just replace img URLs with inline base64
				if (!webHtml) {
					// --pdf-only mode: try reading from existing doc.json, otherwise render
					const jsonPath = resolve(outDir, 'doc.json');
					if (existsSync(jsonPath)) {
						const doc = JSON.parse(readFileSync(jsonPath, 'utf-8'));
						webHtml = doc.html;
					} else {
						const webAssetBase = `/api/project/${slug}/architecture-doc/asset`;
						const result = await renderArchDoc(markdown, webAssetBase);
						webHtml = result.html;
					}
				}

				const htmlWithInlineSvgs = inlineImagesFromWebHtml(webHtml!, docDir);
				const fullHtml = buildPdfHtml(htmlWithInlineSvgs, title);

				const page = await browser.newPage();
				// Block external font requests — use system fallback fonts for PDF
				await page.setRequestInterception(true);
				page.on('request', (req) => {
					const resourceType = req.resourceType();
					const url = req.url();
					if (
						url.startsWith('https://fonts.googleapis.com') || 
						url.startsWith('https://fonts.gstatic.com') ||
						(['image', 'media', 'font'].includes(resourceType) && !url.startsWith('data:'))
					) {
						req.abort();
					} else {
						req.continue();
					}
				});
				await page.setContent(fullHtml, { waitUntil: 'load', timeout: 120_000 });

				const pdfBuffer = await page.pdf({
					format: 'A4',
					printBackground: true,
					timeout: 300_000, // 5 minutes — large docs with many inline SVGs
					margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' }
				});

				writeFileSync(resolve(outDir, 'doc.pdf'), pdfBuffer);
				await page.close();

				process.stdout.write(` | PDF OK\n`);
			} catch (e: any) {
				process.stdout.write(` | PDF FAILED: ${e.message}\n`);
				errors.push({ slug, phase: 'pdf', error: e.message });
			}
		} else if (!pdfOnly) {
			process.stdout.write('\n');
		}
	}

	if (browser) {
		await browser.close();
	}

	const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
	console.log(`\nDone in ${elapsed}s`);

	if (errors.length > 0) {
		console.log(`\n${errors.length} error(s):`);
		for (const e of errors) {
			console.log(`  ${e.slug} [${e.phase}]: ${e.error}`);
		}
		process.exit(1);
	}
}

main().catch((e) => {
	console.error('Fatal error:', e);
	process.exit(1);
});
