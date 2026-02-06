import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { db } from '$lib/server/db/index.js';
import { projects } from '$lib/server/db/schema.js';
import { eq } from 'drizzle-orm';
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { renderArchDoc } from '$lib/markdown.js';
import { buildPdfHtml } from '$lib/pdf-html.js';
import puppeteer from 'puppeteer';

export const GET: RequestHandler = async ({ params, url }) => {
	const project = db
		.select()
		.from(projects)
		.where(eq(projects.slug, params.slug))
		.get();

	if (!project) error(404, 'Project not found');
	if (!project.architectureDocPath) error(404, 'No architecture document for this project');

	// Check if user wants inline view or download
	const disposition = url.searchParams.get('disposition') === 'inline' ? 'inline' : 'attachment';
	const filename = `${params.slug}-architecture.pdf`;

	// Try pre-generated PDF first
	const generatedPdfPath = resolve('../data/generated-docs', params.slug, 'doc.pdf');
	if (existsSync(generatedPdfPath)) {
		try {
			const pdfBuffer = readFileSync(generatedPdfPath);
			return new Response(pdfBuffer, {
				headers: {
					'Content-Type': 'application/pdf',
					'Content-Disposition': `${disposition}; filename="${filename}"`,
					'Cache-Control': 'public, max-age=86400'
				}
			});
		} catch {
			// Fall through to on-the-fly generation
		}
	}

	// Fallback: generate on-the-fly (slow)
	const docPath = resolve(`../data/${project.architectureDocPath}`);
	const docDir = dirname(docPath);

	let markdown: string;
	try {
		markdown = readFileSync(docPath, 'utf-8');
	} catch {
		error(404, 'Architecture document file not found');
	}

	// Extract title from first H1
	const titleMatch = markdown.match(/^#\s+(.+)$/m);
	const title = titleMatch ? titleMatch[1] : project.name;

	// Render markdown with a dummy base URL — we'll replace img src with inline data URIs
	const ASSET_PLACEHOLDER = '/__pdf_asset__';
	const { html } = await renderArchDoc(markdown, ASSET_PLACEHOLDER);

	// Replace asset URLs with inline base64-encoded SVGs
	const htmlWithInlineSvgs = html.replace(
		/<img\s+src="\/__pdf_asset__\?path=([^"]+)"\s+alt="([^"]*)"\s*\/?>/g,
		(_match, encodedPath, alt) => {
			const assetPath = decodeURIComponent(encodedPath);
			const fullPath = resolve(docDir, assetPath);

			// Security: ensure path stays within docDir
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

	// Build full HTML document for Puppeteer
	const fullHtml = buildPdfHtml(htmlWithInlineSvgs, title);

	// Generate PDF with Puppeteer
	let pdfBuffer: Uint8Array;
	let browser;
	try {
		browser = await puppeteer.launch({
			headless: true,
			args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
		});
		const page = await browser.newPage();
		await page.setContent(fullHtml, { waitUntil: 'networkidle0', timeout: 60000 });
		pdfBuffer = await page.pdf({
			format: 'A4',
			printBackground: true,
			margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' }
		});
	} catch (e) {
		console.error('PDF generation failed:', e);
		error(500, 'Failed to generate PDF');
	} finally {
		if (browser) await browser.close();
	}

	return new Response(new Uint8Array(pdfBuffer), {
		headers: {
			'Content-Type': 'application/pdf',
			'Content-Disposition': `${disposition}; filename="${filename}"`,
			'Cache-Control': 'private, max-age=300'
		}
	});
};
